# AnytimeAcquisition
Train a PFN to become an acquisition function trained to optimize anytime performance (and regret).



# TODOs.
* Set up a fork of ifbo. 
* include it via [git submodules](https://git-scm.com/book/en/v2/Git-Tools-Submodules)
* Check how we can account for the final regret as a secondary objective (or normalize); this should make the transformer also reason about what is possible performance-wise.
i.e. similar to the $\alpha$-PFN, we can attempt to calculate the ground truth loss -- or do a proxy over it by all of the seen samples and use that as a normalization on the auc somehow. (i.e. the fixed lower bound cost that we will need to inquire)

## Synthetically prove that with this, we can generate some AUC optimizer.
To avoid all of the moving parts and just show that conceptually this could work, we can make the following POC: 

1. Take a tiny scaled transformer and adjust its input and output projections to accept three inputs and predict two outputs
2. Have a 3-dimensional function, where we have (x, fidelity, y), and we want to optimize the AUC of (fidelity, y) given the hidden x dimension. 
3. Calculate the AUC and sample locally and globally point,s and do the MSE style target-pull on all sequence elements. 
4. Train until convergence and plot the final trajectory the transformer chose.

The idea is that for the auc, we calculate the contours between any two points that the transformer chooses (sequentially) and the integral of the slices of the function in between them, to make it actually stay confined to the surface. 
If we have a fixed start point and end point on a fixed problem, then the transformer will learn to become the **laziest climber** and, during its overfitting to the surface, learn an ideal climbing trajectory.
This basically is a Search problem.

Consider  [BBOB functions ](https://hub.optuna.org/benchmarks/bbob/) or some random shapes from [mathworks](https://ch.mathworks.com/help/symbolic/fsurf.html) 
<!---
f = 3*(1-x)^2*exp(-(x^2)-(y+1)^2)... 
   - 10*(x/5 - x^3 - y^5)*exp(-x^2-y^2)... 
   - 1/3*exp(-(x+1)^2 - y^2)
-->

The integral can be approximated with intermediate points to avoid skipping hills. Also the AUC should implicitly penalize the overall distance, because the longer we walk, the more we need to integrate.

Maybe we can simplify the problem further by simply collecting random sequences and having the transformer converge within the $\epsilon$ ball of possible alterations. Here we can also sample extensively and find the best posssible solution  within the set of trajectories and check it against what the transformer came up with in terms of auc. Also, check against the reference AUC value of the original sequence. 
Fixing the set of trajectories in the dataloader will avoid the generative process altogether and simplify the problem even further. 


We can extend the synthetic case by trying out the same task with different initializations and by having a parametrized family of problems. (e.g., some scaling factors for sine/cosine functions and see how the transformer performs within seen or on unseen parameter ranges. This way, we can evaluate the generalization in prior!


## Architecture: 
1. Dirichlet BNN prior.
2. Frozen first half of the ifbo checkpoint (at best, Lora finetuning allowed). This is our pretrained feature extractor.
3. Make the architecture causal. (which it conceptually was trained to anyway) --> Verify that the change is minimal. How does it affect the predictions of the trained model?
4. Have a residual connection from the input to the second half of the architecture to obtain complete information.
5. Change the output layer to predict an HPs + fidelity dimensional output.
6. Calculate the AUC based on the difference from the best ACC.
7. sample locally and globally and out of basket around the points from the dirichlet example (causally) (including fidelity dim) evaluate the y values in the prior and calulate the updated AUC.
8. Now that we have the AUC for each alternative rollout, we can use MSE to pull the HP output from a position towards the best alternative given the context.

Advantages: 
+ The AUC reflects both the fidelity spent and the cost associated with it, as well as the performance gain.
+ We exploit a strong feature extractor for learning curve projection
+ We basically have some Monte Carlo rollouts
+ We can differentiate through the otherwise nondifferentiable AUC
+ This is a single forward acquisition function purely prior trained


## Generative approach to diverge from the prior's limitations.
Since the random policy implied by the sequentially read Dirichlet prior will ultimately be limited in its ability to provide good solutions, 
We should aim to diverge from this "collecting" policy and instead take a recent checkpoint of the model and use it as a generative policy -- 
We can do this because ultimately it is bound by the truth contained in the prior, and we have access to it. 
Given that we fill a replay buffer with those sequences generated according to the local/global/basket sampled optimization, we can optimize over them with known targets.
To boost the capability, we can take any previously generated sequence and given the prior seed that instantiated the BNN simply still have access to the ground truth -- but now actually make 
actual rollouts at certain inception points of the sequence; i.e. if we had conditioned on parts of the sequence, we can determine a set of points (stick breaking?) where we want to produce rollouts and optimize. 
Given these new experiences, we can train the model again. (Same process with task alteration, or parallel process with checkpoint & buffer communication)


## Technical details. 
+ Huggingface Transformer implementation with
+ custom beam search and
+ key value caching.
+ Maybe accessing the attention weights on the full sequence.

## Sorting out kinks: 
Key theoretical aspects
+ how we evaluate the advantage of a single token**, once we have permuted the sequence. 
+ How do we want to permute: RND / attention weight RND / RND with rejection sampling based on AUC? / beam search? 

