# Problem setting

Shared background for everything in `docs/ROADMAP.md` — written once here so
the roadmap and future design docs can reference it (`§PS.n`) instead of
re-explaining it. If this file and the actual code
(`src/anytimeacquisition/models/pfn.py`, `models/bar_distribution.py`,
`priors/bnn.py`, `trainer/pfn_trainer.py`) ever disagree, the code is
authoritative — update this file, don't trust it blindly.

---

## PS.1 The PFN: what it is, how it's trained

A **Prior-data Fitted Network** (PFN) is a transformer trained to do
in-context Bayesian inference: given a *prior* — any distribution over
functions that can be cheaply sampled from — it learns to map a set of
observed `(x, y)` pairs to a calibrated posterior predictive distribution
(PPD) at a query point, in a single forward pass, no explicit inference
procedure at deployment time.

**Our prior is a sampled BNN** (`priors/bnn.py`): each draw is an
independently-sampled random-architecture tanh MLP — random depth, random
width, width-compensated init scale — which *is* the function to be
minimized. Depth, width, and weights are the latent variables of the prior;
sampling one draw is sampling one black-box optimization problem.

**Training objective.** For each prior draw, split its domain into a
*train* set (the observed context) and a *test* set (query points). The
model reads `(x_train, y_train)` and predicts, for each `x_test`, a
distribution over `y_test` — output as **logits over a fixed set of bins**
(a piecewise-uniform "bar distribution" over the support, not a Gaussian
mean/variance), trained by cross-entropy / NLL against the realized `y_test`
values. Averaged over many fresh prior draws, minimizing this loss is
*provably* equivalent to learning the true Bayesian posterior predictive
under the prior (the "PFN theorem") — the network is doing amortized
Bayesian inference by supervised learning, not by explicit inference at
deployment time.

**Architecture — why the attention pattern is the whole point, not an
implementation detail:**
- **Train tokens self-attend to train tokens, bidirectionally.** This is
  what makes the train-token representations a *permutation-invariant
  summary* of the observed context — exactly what a Bayesian posterior
  should be (order of observation carries no information a correct
  posterior should depend on).
- **Test (query) tokens cross-attend to train tokens only** — never to each
  other, never to themselves. No test-test leakage: every query's
  prediction is conditionally independent given the context, which is
  required for the output to be a valid predictive distribution
  point-by-point, and it's what makes a batch of query tokens a *parallel
  acquisition-function evaluator over an arbitrary candidate set* — score
  any number of candidates in one forward pass, and adding or removing
  candidates doesn't change any other candidate's score.
- This happens at **every layer**, not just the output — so a query token's
  final-layer representation, before the last projection to bar-distribution
  logits, is already a rich, context-aware summary of "this candidate, in
  light of everything observed so far." (Relevant later — see `§PS.3`.)

**The train tokens, taken together, are the sufficient state description.**
There is no separate "belief vector" the model computes and then consults —
the state *is* the set of train-token representations, and any query
against it is answered by cross-attending into exactly that set. This
matters mechanically: **the train context uses bidirectional attention, so
any change to the observed set (a new `(x, y)` pair) requires a fresh
forward pass over the whole context.** Nothing about a previous decision
step's computation can be incrementally reused once the context changes —
see `§PS.4` for what this does and doesn't cost.

**In-context learning, not fine-tuning.** At deployment, the frozen network
sees a brand-new problem's `(x_train, y_train)` and produces a valid PPD for
it immediately — no gradient steps, no per-problem adaptation. This is only
true because training explicitly optimized for it (fresh prior draws every
step, per `trainer/pfn_trainer.py`), and it's the property this whole
project's downstream design depends on: a frozen model that's *already* a
valid, calibrated encoder of "what do I currently believe," for any context
drawn from the same prior family.

## PS.2 What a PPD is a sufficient statistic for

The PPD at a candidate point is not just "the model's guess" — it's the
sufficient statistic every classical acquisition function is a closed-form
function of: EI, PI, UCB are all deterministic transforms of the predictive
distribution at a point (mean/variance for a Gaussian surrogate; a sum over
bucket probabilities for a bar distribution — see `models/bar_distribution.py`'s
`ei`/`pi`). Anything trained to consume the PPD directly starts with access
to everything a hand-derived acquisition function has access to, before
learning anything else.

## PS.3 Architectural inspiration: reading a frozen backbone's internal state (π0.5)

**π₀.₅** (Physical Intelligence, [arXiv:2504.16054](https://arxiv.org/abs/2504.16054),
code at [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi))
is a Vision-Language-Action (VLA) model: a frozen(-ish) vision-language
backbone produces a rich internal representation, and a separate **action
expert** — its own weights, not part of the backbone — reads that
representation via cross-attention into the backbone's own keys/values at
multiple layers, rather than consuming only the backbone's final text
output. The backbone's forward pass is untouched; the expert is a parallel
reader with its own parameters.

The relevance here is architectural, not domain: it's a template for how a
small, separately-trained module can extract a rich state representation
from a larger frozen model **for a downstream task the frozen model was
never trained to directly support** — in our case, scoring candidate query
points for *decision-making* value, not just predicting their own `y`
distribution. The PFN's own query tokens already cross-attend into the
frozen context at every layer (`§PS.1`); the open architectural question
this project inherits from π0.5's example is *how much* of that per-layer
computation a downstream scoring head should read directly, versus only
consuming the PFN's own final, task-specific (NLL-optimized) output — see
`docs/ROADMAP.md` for where this project lands on that question and why.

## PS.4 Compute shape: what's expensive, what isn't, and why

Three genuinely different costs get conflated if this isn't kept precise
(see `docs/ROADMAP.md`'s own build-cost discussion for the numbers):

- **Within one decision, across candidates:** cheap. One batched forward
  scores an entire candidate pool at once (query tokens never attend to
  each other), so evaluating `C` candidates costs one forward call, not `C`.
- **Within one episode, across steps:** here is where bidirectional
  attention actually costs something — every new observation invalidates
  the previous forward pass's computation entirely (`§PS.1`), so an episode
  of length `B` costs `O(B²)` token-work, not `O(B)`. In absolute terms
  this is still small for realistic `B` (a few thousand tokens for `B~64`),
  and the candidate count `C`, not the recompute, tends to dominate cost in
  practice once `C` is a meaningful fraction of `B`.
- **Across independent episodes/functions, and across gradient steps:**
  cheap and fully parallel — different episodes (or different states pulled
  from a replay buffer) share no computation with each other regardless of
  attention pattern, so this axis batches as wide as memory allows.
