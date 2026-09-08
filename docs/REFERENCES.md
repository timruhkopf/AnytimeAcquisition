# References

Bibliography for `docs/ROADMAP.md`/`docs/PROBLEM_SETTING.md`. Each entry:
title, authors/venue, links, what it does, and (where filled in) how it
differs from this project's own approach — that last field is being added
incrementally as our own design solidifies, not exhaustively up front; a
`_TBD_` there means genuinely not yet worked out, not "no difference."

Add entries here rather than re-explaining a paper inline in the roadmap.

---

## Surrogate literature

### PFNs4BO: In-Context Learning for Bayesian Optimization
Müller, Feurer, Hollmann, Hutter — ICML 2023.
[arXiv:2305.17535](https://arxiv.org/abs/2305.17535) ·
[github.com/automl/PFNs4BO](https://github.com/automl/PFNs4BO)

Uses a Prior-data Fitted Network as a drop-in surrogate for Bayesian
optimization: trained once via in-context learning on a chosen prior
(a naive GP, an advanced GP, or a BNN), then deployed frozen to produce a
calibrated posterior predictive at any query point in one forward pass, no
per-problem fitting. Also shows how to inject side information (hints about
optimum location, irrelevant-dimension masking) and demonstrates learning a
non-myopic acquisition function on top of the PPD. This project's own
`models/pfn.py`/`priors/bnn.py` are a from-scratch implementation of the
same idea (BNN prior specifically), not a fork of this codebase.

**How it differs from our idea:** _TBD_.

### In-Context Freeze-Thaw Bayesian Optimization for Hyperparameter Optimization (ifBO)
Rakotoarison, Wistuba, Franke, ... — ICML 2024.
[arXiv:2404.16795](https://arxiv.org/abs/2404.16795) ·
[github.com/automl/ifBO](https://github.com/automl/ifBO)

Extends the PFN-as-surrogate idea to grey-box, freeze-thaw HPO: a PFN
(FT-PFN) trained to extrapolate partial learning curves in-context, letting
the optimizer allocate scarce training budget incrementally across many
partially-trained configurations rather than committing to full training
runs. Reports 10-100x faster and more accurate predictions than the deep
GP/deep-ensemble surrogates used in prior freeze-thaw BO work, paired with a
randomized-horizon acquisition mechanism (MFPI-random).

**How it differs from our idea:** _TBD_.

### TabPFN: a transformer that solves small tabular classification problems in a second
Hollmann, Müller, Eggensperger, Hutter — ICLR 2023 (original); TabPFN v2,
Nature 2025.
[arXiv:2207.01848](https://arxiv.org/abs/2207.01848) ·
[github.com/PriorLabs/tabpfn](https://github.com/PriorLabs/tabpfn)

The PFN idea applied to general tabular classification/regression rather
than BO specifically: pretrained once on millions of synthetic datasets
(sampled from a prior over data-generating processes, including
structural-causal-model-like priors), then deployed zero-shot — the entire
downstream training set is the in-context prompt, no gradient steps at
deployment. The foundational demonstration that PFN-style in-context
learning scales to genuinely useful, general-purpose prediction, not just a
BO-specific surrogate; PFNs4BO and ifBO both build on this line directly.

**How it differs from our idea:** _TBD_.

---

## MetaBO / NAP / FSAF — direct precedent for this project's own approach

Named specifically in the design discussion `docs/ROADMAP.md` is based on
as the reason to score a discrete candidate set with RL rather than output
a continuous action (`docs/ROADMAP.md` §1.1) — kept as their own section
rather than folded into the general acquisition-strategy list below,
since these are the direct precedent, not just adjacent work.

### Meta-Learning Acquisition Functions for Transfer Learning in Bayesian Optimization (MetaBO)
Volpp, Fröhlich, Fischer, Doerr, Falkner, Hutter, Daniel — ICLR 2020.
[arXiv:1904.02642](https://arxiv.org/abs/1904.02642)

Trains an acquisition function with reinforcement learning across a
distribution of related tasks (fixed GP surrogate), so the learned
acquisition function picks up transferable structure in the objective
functions rather than using a fixed hand-derived formula. The policy scores
a **discrete candidate set** rather than outputting a continuous action
directly — one of the papers (with NAP/FSAF, below) that established this
is the parameterization that actually trains for learned acquisition
functions, which is why this project adopts it too.

**How it differs from our idea:** _TBD_.

### Reinforced Few-Shot Acquisition Function Learning for Bayesian Optimization (FSAF)
Hsieh, Hsieh, Liu — NeurIPS 2021.
[proceedings.neurips.cc PDF](https://proceedings.neurips.cc/paper_files/paper/2021/file/3fab5890d8113d0b5a4178201dc842ad-Paper.pdf) ·
[arXiv:2106.04335](https://arxiv.org/abs/2106.04335)

Addresses the fact that no single hand-derived acquisition function is
best across all problem types: learns a **distribution** of Q-networks (a
Bayesian DQN variant) as acquisition functions, meta-trained (MAML-style,
few-shot) across a family of tasks with a KL-regularization term to avoid
overfitting to any one, and explicitly uses demonstration trajectories from
existing classical acquisition functions as priors/warm-starting signal —
the same shape of idea as this project's own `LogEI` distillation warm
start (`docs/ROADMAP.md` §3 M2), independently arrived at. Reports being
agnostic to input dimension and candidate-set cardinality. Notably
DQN-based, i.e. value-based with a max over the candidate set — the exact
pattern `docs/ROADMAP.md` §1.3 explicitly avoids (overestimation bias
scaling with the number of actions maxed over); worth checking whether
their reported results show any symptom of that when this section's "how
it differs" gets filled in properly.

**How it differs from our idea:** _TBD_.

### End-to-End Meta-Bayesian Optimisation with Transformer Neural Processes (NAP)
Maraval, Zimmer, Grosnit, Bou Ammar — NeurIPS 2023.
[arXiv:2305.15930](https://arxiv.org/abs/2305.15930)

Jointly trains the surrogate *and* the acquisition function end-to-end — a
transformer neural process (the "Neural Acquisition Process") that predicts
both a posterior-like distribution and a distribution over candidate
actions, trained with reinforcement learning to handle the lack of labeled
acquisition data. Notably: training a transformer neural process with RL
from scratch was hard enough that the authors needed to add a **supervised
auxiliary loss** to keep part of the network a valid probabilistic model as
an inductive bias — direct motivation for this project's own `§1.4`
decision to freeze an *already*-valid, separately-pretrained PFN rather
than train the equivalent jointly from scratch.

**How it differs from our idea:** _TBD_.

---

## Acquisition-strategy literature — adjacent, not direct precedent

### Reinforced In-Context Black-Box Optimization (RIBBO)
Song, Gao, Xue, Wu, Li, Hao, Zhang, Qian — Nanjing University / Huawei
Noah's Ark Lab, 2024.
[arXiv:2402.17423](https://arxiv.org/abs/2402.17423)

Learns a black-box optimization algorithm end-to-end with **no RL**: a
causal transformer is trained offline, purely supervised, on optimization
trajectories generated by a portfolio of existing behavior algorithms, each
augmented with a *regret-to-go* conditioning token (analogous to
return-to-go in a decision transformer). A Hindsight Regret Relabelling
scheme keeps those tokens valid at inference. At deployment, conditioning on
a strong target regret makes the model imitate (and in their experiments,
exceed) the best algorithm in its training portfolio, in-context, without
manual algorithm selection. Directly relevant here as a candidate v1
diagnostic — see `docs/ROADMAP.md` M0 — since this project's own `Ḡ_t∈[0,1]`
return is already in exactly the return-to-go form this approach needs.

**How it differs from our idea:** _TBD_.

### PABBO: Preferential Amortized Black-Box Optimization
Zhang, Huang, Kaski, Martinelli, 2025.
[arXiv:2503.00924](https://arxiv.org/abs/2503.00924)

Fully amortizes *preferential* Bayesian optimization (learning from pairwise
comparisons between candidate designs rather than direct function values,
motivated by interactive human-in-the-loop settings) — both the surrogate
and the acquisition function are replaced by a transformer neural process,
trained with reinforcement learning plus auxiliary losses, bypassing
per-step GP inference entirely. Reports orders-of-magnitude faster
deployment than classical preferential BO with comparable or better
accuracy. Closest of this section's three papers to "amortize both the
surrogate and the acquisition function jointly, train the whole thing with
RL" — differs from PFNs4BO-style work in optimizing the acquisition
function, not just the surrogate, end-to-end.

**How it differs from our idea:** _TBD_.

### Direct Regret Optimization in Bayesian Optimization
Zhang, Chen, 2025.
[arXiv:2507.06529](https://arxiv.org/abs/2507.06529)

Trains a decision transformer offline on simulated trajectories from an
ensemble of GPs with varied hyperparameters, jointly distilling a model and
a non-myopic acquisition strategy aimed directly at minimizing multi-step
regret, rather than composing a fixed acquisition formula on top of a
separately-fit surrogate. Uses a dense-training/sparse-refinement split:
heavy offline training on abundant simulated trajectories, light online
adaptation on the limited real evaluations actually available. Reports
lower simple regret than classical BO baselines, particularly at higher
dimension and under observation noise — directly relevant to this project's
own concern about behavior at `d` up to 18.

**How it differs from our idea:** _TBD_.

---

## Architecture / cross-attention design inspiration

### π₀.₅: a Vision-Language-Action Model with Open-World Generalization
Physical Intelligence (Black, Brown, et al.), 2025.
[arXiv:2504.16054](https://arxiv.org/abs/2504.16054) ·
[github.com/Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)

Not a BO paper — included for architectural inspiration only. See `docs/PROBLEM_SETTING.md`
§PS.3 for the specific mechanism (a separate action-expert module reading a
frozen backbone's keys/values via its own cross-attention weights) and why
it's relevant to how a downstream scoring head reads the PFN's internal
state.

**How it differs from our idea:** not applicable — inspiration for a
mechanism, not a comparable end-to-end approach.
