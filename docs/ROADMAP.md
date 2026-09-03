# Roadmap

This project builds a learned, in-context acquisition function for Bayesian
optimization: given a growing set of observed `(x, y)` pairs from an unknown
function, choose the next `x` to query. The learned piece is a policy —
`ActionHead` — that reads a frozen, pretrained transformer's representation
of the observed data and outputs a next point, no explicit acquisition
formula (EI/UCB/PI) hand-derived or hard-coded.

Two lenses organize everything below, deliberately kept separate because
they answer different questions: **VLA architecture** describes what gets
built and how it behaves once deployed; **ground-truth privileged search**
describes how its training labels get generated, using information a
deployed policy never has. They meet at exactly one point — receding-horizon
replanning — noted where it comes up in each.

Current status, what's validated vs. not, and the prioritized next
experiments live in `docs/MILESTONES.md`, kept separate from this file on
purpose: this file is about *why* the system is shaped the way it is, that
file is about *where it currently stands*.

## Lens 1 — VLA architecture

The system is structured the way a Vision-Language-Action model is
structured: a large, frozen representation-producing backbone, and a smaller
"expert" head that reads that backbone's internal state to produce actions.
Concretely, here that maps to:

- **The PFN (`models/pfn.py`) is the backbone.** A Prior-Fitted Network,
  pretrained (`trainer/pfn_trainer.py`) to take a set of observed `(x, y)`
  pairs and produce a calibrated predictive distribution (a bar/histogram
  distribution, `models/bar_distribution.py`) at any query point —
  permutation-invariant over the observations, the same shape of object a
  VLM's vision-language representation is to its own downstream action
  expert.
- **The PFN is frozen, by reasoned choice, not by default.** VLA models like
  pi0.5 fine-tune their backbone because there's a real domain gap between
  broad web-scale pretraining and precise low-level robot control. There's
  no equivalent gap here — the PFN was purpose-built from scratch for
  exactly the representation `ActionHead` consumes, so the transfer-learning
  argument for fine-tuning doesn't transplant. Freezing is also
  **load-bearing for training-label generation**, not just an architectural
  preference: the privileged search (Lens 2) does gradient descent directly
  against the PFN's own predictive distribution as a trusted, fixed oracle.
  If training pressure on `ActionHead` could reshape the PFN, a policy could
  learn to reshape the very landscape its own labels are scored against — a
  reward-hacking-shaped failure specific to this pipeline, not a generic
  fine-tuning caution.
- **`ActionHead` (`models/action_head.py`) is the action-expert.** Currently
  a single-shot cross-attention readout: one forward pass, its own per-layer
  K/V projection into the PFN's per-layer hidden states (`return_hidden=True`
  — a plain list of full hidden states, not an incremental/cacheable KV
  cache), producing one Beta distribution per input dimension — one next
  point, no lookahead built in. That per-layer-own-weights structure is
  mechanically forced by freezing (frozen weights can't be trained, so the
  head necessarily needs its own parameters at every layer it taps) — it
  incidentally resembles pi0's own separate-expert-weights pattern, for a
  different underlying reason than pi0's.
- **Whether that single-shot shape is sufficient is open, again — on new
  grounds.** A flow-matching, chunked head (reusing pi0.5's rectified-flow
  mechanism: `x_t = t·noise + (1-t)·data`, constant target velocity `u_t =
  noise - data`, `MSE(v_t, u_t)` loss, Euler-integration sampling) was
  considered and declined once already, but for a different reason
  (representing a *multimodal single point*, motivated by many human
  demonstrators — not applicable here, where every label comes from one
  deterministic oracle call). The justification worth taking seriously now
  is different: genuine multi-step *chunked planning*, matching how Lens 2's
  k-step privileged search already produces short plans and keeps only the
  first point. Concrete reuse assessment (`github.com/Physical-Intelligence/openpi`,
  Apache 2.0; PyTorch port at `huggingface/lerobot`, also Apache 2.0):
  the flow-matching loss/sampler are small, generic, and not
  PaliGemma-specific, so they port cleanly; the conditioning interface
  (`gemma.Module`) takes a list of already-embedded continuous token
  streams and is genuinely backbone-agnostic — only `Pi0.embed_prefix()`
  (SigLIP + text tokenizer) is VLM-specific and would be discarded
  entirely; `action_horizon` (chunk length, their default 50) and
  `num_steps` (denoising iterations, their default 10) are decoupled knobs,
  not the same number. pi0's actual attention mechanism is *not* literally
  "cross-attend into a frozen KV cache" — it's one joint self-attention op
  over concatenated streams with per-stream expert weights and a
  block-causal mask. Our current `ActionHead` is architecturally a cleaner
  match to "cross-attend into a frozen backbone" than pi0 itself is
  (genuine cross-attention, not joint self-attention) — so adopting flow
  matching should keep our own cross-attention structure and only add
  iterative denoising + multi-token chunk output on top of it, not port
  pi0's joint-attention pattern. The PFN's lack of an incremental/cacheable
  state means hidden-state computation must be decoupled from head
  execution before denoising iterations are cheap — a real prerequisite,
  not a style choice.
- **Deployment is causal, receding-horizon control.** At decision time,
  `ActionHead` sees only what's actually been observed — no privileged
  information, ever. If/when the head plans a multi-step chunk, only a
  prefix gets executed before replanning from the real, newly-observed
  state — the same discipline real VLA inference already uses, and exactly
  where this lens meets Lens 2's own k-step planning below.

## Lens 2 — ground-truth privileged search

This is the training-label-generation side, and its central discipline is
strict: **privileged, train-time-only access to the true environment is used
exclusively to generate labels, and never leaks into what a policy actually
observes.** `priors/bnn.py`'s `BNNPrior.evaluate(..., noise=False)` is the
privileged oracle — an exact, differentiable, noise-free evaluation of the
true function a real rollout only ever samples noisily.

- **Exploit search (`search/exploit.py`).** Direct gradient descent on the
  true surface, multistart from the current incumbent plus random restarts,
  toward a better point than what's already been observed — pure
  ground-truth optimization. No model in the loop for *choosing* where to
  search, only (downstream) for scoring whether the correction is worth
  imitating.
- **Explore search (`search/explore.py`).** Gradient descent on a candidate
  query, teacher-forced through the true environment fresh every step (never
  a stale or self-predicted value), scored by the frozen PFN's own
  calibration — NLL against known true values at a fixed, privileged test
  set, not entropy (entropy can be gamed by confident wrongness; NLL against
  a known answer can't). Each test point is weighted by how much genuine,
  ground-truth improvement it still represents over the current incumbent —
  points the incumbent already beats contribute nothing, since resolving
  uncertainty about them can't reduce regret.
- **k-step generalization — a first-class part of the design, not a
  side experiment.** Both branches above are one-step-lookahead special
  cases of the same underlying idea: jointly optimize a short *sequence* of
  future points against the privileged, ground-truth objective, keep only
  the first point as the actual label, discard the rest, and replan fresh
  next step (receding horizon — exactly Lens 1's deployment framing, met
  here from the label-generation side). This is deliberately kept as a
  planning tool that produces better *first moves* by looking ahead, not as
  something that ever gets executed multi-step blind.

  One finding from developing this is load-bearing enough to state as a
  standing rule, not a footnote: **a joint multi-point plan's score is the
  *team's* value, not any single point's own.** The PFN reads context as a
  permutation-invariant set — it has no notion of "this point arrived before
  that one" — so a plan's reported objective value cannot be decomposed back
  into per-point credit after the fact (the same problem Shapley values
  exist to solve for jointly-produced value in cooperative game theory). It
  is valid to use a joint plan's score to *select* the first point (the
  standard model-predictive-control argument: that point's job is to lead
  somewhere a good continuation exists, not to deliver the whole plan's
  value alone). It is **not** valid to report or trust that joint score as
  what the selected point is actually worth. The fix, non-negotiable
  wherever k-step search is used: after selecting a point via the joint
  objective, always re-score it *alone*, with an independent single-point
  forward pass, before treating that number as its real contribution.
- **BC/EXIT: training on top of the real model's own rollouts, not static
  offline data.** Roll out the current policy causally (no privilege at
  decision time; round 0 uses a random policy as a bootstrap, later rounds
  phase in the real, still-imperfect `ActionHead` via DAgger-style mixing).
  At every step, invoke the privileged search (Lens 2) for what should have
  been done from that exact context, and train `ActionHead` via behavioral
  cloning toward that correction. Repeat across rounds — the policy's own
  rollouts get better, keeping the training distribution close to what the
  policy will actually see at deployment, rather than training once against
  a fixed, increasingly-stale dataset.
- **Not every privileged correction is a trustworthy label, and that has to
  be checked, not assumed.** A correction's own search objective improving
  is not the same claim as "this correction reduces true regret" — the two
  are correlated but not interchangeable, and the correlation itself is not
  fixed: measured directly this project's own way (`greedy_regret`,
  `pipelines/explore_search_playground.py`), it varies by checkpoint and
  scale rather than holding as a constant property of the mechanism. See
  `docs/MILESTONES.md` for the actual numbers and what they currently imply
  about which fixes are worth doing first.

## Where this leaves the two lenses

Lens 1 asks "what does the deployed policy look like and how capable is
it." Lens 2 asks "how do we generate a trustworthy signal to train it
toward." Neither is complete on its own — a more capable architecture
trained on unreliable labels doesn't obviously beat a simple architecture
trained on validated ones, and a perfectly-validated label pipeline still
needs an architecture actually capable of representing what it's taught.
`docs/MILESTONES.md` tracks both halves separately for exactly this reason —
component-level status, not just an end-to-end pass/fail — and prioritizes
what to fix next accordingly.
