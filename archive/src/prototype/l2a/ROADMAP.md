# Roadmap

## Abstract Goal

We want to learn how to optimize the probability space of a pre-trained PFN in order to get an emerging acqusition
function without the need to do rlhf, but go fully supervised

---

## Architectural details:

* log regret as loss signal (Anytime regret) for the acquisition function learning.
* \[ACQ\] token appended to the context, allowed to cross attend to all train and test tokens
* \[ACQ\] token should be learnable initially / could be random token
* test tokens: LHD design to get some global state information wrt. the probability space of the task
  (will fail if a user actually has LHD design, but we can start with it). Intuition: the test tokens carry information
  about the probability space, since they are unkonwn, and they will allow us to extrapolate this information to any
  point of interest in the probability space for any given task.
* use only cross attention (since we have a single acquisition token). This will ease the learning and reduces the
  number of parameters.
* Causality: Since we want a model that can at any point in time during optimization process propose the next point, we
  need to make sure, that the train tokens are causally masked, the test tokens causally attend to the train tokens, and
  the \[ACQ\] token can attend to all tokens causally -- meaning, this one token is computed autoregressively for any
  horizon. #Consider: maybe we could just avoid the acq token all togther (loosing recursive abilities down the line),
  and have the model predict at every train output position the next point it wants to query! --NO: this is incompatible
  with the frozen PFN's forward pass - since it dictates, how the tokens should look like
* Once we have forwarded the \[ACQ\] token and collected its predicted location, we need to find its label: Since we
  have access to the prior, and can evaluate it, we can simply sample $k$ locations in the epsilon ball (N(\[ACQ\],
  eps)) around the proposed point and evaluate the prior in batch mode for all of them, then use the best point as ideal
  label and weight the position's loss with the log regret. #FIXME: Supervised Learning for Local Descent doesn't this
  starve the model of a learning signal for exploration? -- Not a Bayesian optimizer it just encourages it to be greedy
  on the acquisition function, not to tap into high uncertainty regions. - weigh each sequence by the minimum regret
  found over T encouraging optimization on the batch level. Maybe this will steer the model towards more exploration
  because only that will maximize the overall expectation! However, this makes for a potentially collapsing training
  dynamic, when the batch basically becomes a single policy (unlikely in continuous space?). One more thing:
  The model learns: "Even though there was no local improvement at step 5, the decision was strategically valuable."
  The Risk: This is essentially a high-variance gradient. To stabilize this, you might need a Baseline (like the average
  min-regret for that prior family) so you only amplify "above-average" trajectories.
* eps annealing on the eps ball around the [ACQ] over time: this encourages early exploration and later exploitation for
  the target label collection

---

## Experiments

train on a fixed length (budget 50)

### 1. POC

Preparation: Train a small scale non-causally masked 1d "set" PFN on a simple BNN prior. Then introduce causality for
the train section, and the test section as well as

### 2. Dropping the LHD design for the test tokens

Conceptually, the PFN's train representations should suffice to inform us about the probability space --> because that
the cross attention has learned to do. This is basically a lobotomy on the PFN.
The Lobotomy: If you remove the LHD tokens, the [ACQ] token has to "hallucinate" the uncertainty map solely from the
gaps in the training data. While a powerful Transformer can do this, the LHD tokens act as a "cheat sheet" that makes
the learning much faster

### 3. Ablations

* the eps annealing schedule for the label collection (linear, exponential decay, ...?)
* Trajectory-wide hindsight: the min regret weight for the batch sequence loss: if we weigh the batch item (sequence) by
  the minimum regret found over the T steps, then we encourage the model to find trajectories that have good minima,
  which may encourage exploration more than just being greedy descent acquistition function
* should we use random instances of the MLP in the batch (making it more difficult to train because of lack in
  vectorization over varying model sizes), or should we in one batch have many runs on the same MLP instance? -- Since
  we want to weigh the batch by the minimum regret, this would directly optimize the strategy over many alternatives!
  Conveniently, this also allows us to collect the samples of and the labels in batch mode on one mlp instance.
  By running multiple trajectories on the same $BNN_\theta$ instance, you are performing a mini-evolutionary strategy in
  your batch. The model sees three different ways to optimize the same function—one found the minimum, two didn't. The
  gradient will very clearly point toward the "winning" strategy. (start with this)

**Validation Metrics**:

* Regret curves (log-regret over time) vs rnd search and standard BO (GP with EI acquisition function).
* By competitor heuristics:
  Train a GP on 1d data for every step, then optimize the acquisition function (EI /PI / UCB) to get the next point,
  then measure the distance of the point proposed by the acquisition function to the point proposed by the \[ACQ\]
  token.

---

## Future Directions

### Recursion

Since the \[ACQ\] token is actually an hp, and the prediction of the model is also in the hp domain, we can
actually feed the proposed point back into the model as a new query \[ACQ\] token -- doing so extremely efficient with
KV caching

### Recursion V2

What happens, if in a second forward pass, we would append the proposed point to the test tokens -- this way, we the
model
could refine its belief based on the explicit pfn probability evalution of the proposed point. This probably would
require
to have some random set of points in the test tokens, so that a position other than the LHD is not per se suprising.

### Batch Acquisition Function Optimization

* Cosine Similarity penalty on the last hidden representation to encourage diversity in the batch.
* Causality masking mixed with self attention on the batch tokens allows us to have a "diverse" and coordinated batch
  acquisition function

### Horizon: Exploration Exploitation under Budgeting

The main insight is, that different budgets cause different optimal acquisition functions:
In BO, "Exploration" is just "Exploitation with a long time-horizon."

* adding a sin / cosine based PE for remaining budget on the first cross attention layer (from \[ACQ\] to TRAIN) allows
  the model to learn budget-aware acquisition strategies (a non-stationary policy).
* masking the post budget tokens
* Batch weighing to encourage exploration won't be possible anymore natively, because the long horizon gradients hijack
  the learning signal
* The "Advantage" Weighting (Normalization by Horizon)Instead of weighting by absolute log-regret, you should weight
  by Relative Performance. In Reinforcement Learning, this is known as the Advantage Function.The Mechanic: For every
  task in your batch, you calculate the log-regret. But before applying it as a weight, you subtract a Baseline ($B_H$)
  that is specific to that horizon ($H$).The
  Formula: $Weight = (\text{MinLogRegret}_{actual} - \text{ExpectedMinLogRegret}_H)$.The Result: A 5-step sequence that
  found a "lucky" local minimum gets a higher weight than a 50-step sequence that found a mediocre global minimum. This
  forces the model to maximize its Efficiency-per-Step, regardless of the total budget.
* Simpler:
  Stratified BatchingIf the "Fairness" math gets too noisy, the simplest engineering fix is Stratified Batching.In one
  training batch, all sequences must have the same horizon.You cycle horizons across batches (Batch 1: $T=10$, Batch
  2: $T=50$, etc.).Since every item in the batch is competing on a level playing field, you can safely use your
  Min-Regret Weighting without normalization. The model still learns the "Global Policy" because the weights are
  internally consistent within each forward pass.
* The Final Scrutiny: "Starvation" vs. "Collapse"
  Starvation: If you don't weight, the model becomes a greedy local-searcher (Safe but mediocre).
  Collapse: If you weight poorly, the model only learns one "type" of search (Long-horizon) and fails everywhere else
* Use the Horizon-Specific Baseline. It’s the most mathematically sound way to keep the "Success Weighting" active
  across all budgets. It transforms the training objective into: "Be better than the average optimizer at this specific
  point in time." i.e. calculating a baseline for each horizon (e.g., using a simple random-search average)

### Hypothesis: Greedy on the LHD test tokens

If we have LHD for test tokens, then it would be simpler to just evaluate the points than extrapolating their
probability distributions. Essentially questioning the need for the test tokens of the PFN

### Use the model as a Policy network and perform RLHF on top of it.

* Anytime performance on the log-regret?

### Hypothesis: Train test distribution shift: random sequences do not look like optimizer trajectories!

Batch KV-Cached generation of BO trajectories for a self-improvement loop. How do we ensure the model does not collapse?
This suggests, that we should pre-pend the the acq token int the context, and have the train tokens at the end of the
context to avoid insertations:
\[[\ACQ\], test tokens, train tokens\]

It allows us to grow the history without re-processing the "query" architecture.

### Multi-fidelity optimization:

* adding the fidelity dim to the hp input
* calculating / weighing the log-regret imporvement by the cost delta.
* sample eps ball in cost and locations. The question is whether we get a good policy signal, that can reverse on its
  path dependency?
* We probably want to sample the prolongation of sequences more frequently with higher overall cost!