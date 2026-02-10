"""
You are 100% correct. If we only look at the ground-truth environment's perspective,
the "Area of Possible Improvement" only shrinks when the incumbent changes.
My previous explanation assumed the surrogate's belief was involved; but using only the raw environment values,
the "flashlight" effect doesn't exist.To get the dense signal you are looking for—where the agent is rewarded
for knowledge gain even when the incumbent doesn't change—we have to bridge the gap between the Env's Truth
and the Model's Belief.The Solution: "Pseudo-Volume" via the SurrogateIf we want a reward that moves every time
we sample (even if we don't find a new best), we need to measure the volume under the incumbent as seen through the
eyes of the PFN.Here is the refactored logic:Ceiling: Use the True Incumbent from the Environment (Ground Truth).
Floor: Use the PFN Surrogate's Mean/Quantile (Model Belief).
The Reward: The reduction in the volume between the
True Incumbent and the PFN's Current Prediction. Now, if you sample a "bad" point ($y > y_{incumbent}$), the PFN
updates its belief in that region. The PFN's "Floor" rises toward the true value. Because the floor rose, the volume
between the floor and the ceiling shrank.This provides the dense reward you want. The agent is rewarded for "raising
the floor" (ruling out bad areas) or "lowering the ceiling" (finding new incumbents).
"""


"""
The Advantage of the Surrogate: The surrogate is a "Continuous Function" of your observations. 
Every single $(x, y)$ pair you feed it—even a bad one—warps the surrogate's surface.
The Synergistic Reward: By measuring the volume between the Env's Best and the Model's Mean, 
you create a signal that rewards the agent for reducing its own uncertainty relative to the best known result.
"""

class SurrogateVolumeReward:

    def __init__(self, pfn_model, monitor_sampler, step_penalty=0.01, device='cuda'):
        self.pfn = pfn_model
        self.sample_monitor_points = monitor_sampler
        self.step_penalty = step_penalty
        self.device = device

    def __call__(self, obs_traj, env):
        T, B, _ = obs_traj.shape

        # 1. Get True Incumbents from Env (The Ceiling)
        y_vals = obs_traj[..., 2]  # (T, B)
        incumbents, _ = torch.cummin(y_vals, dim=0)  # (T, B)

        # 2. Get Monitor Points for Volume Estimation
        mon_x = self.sample_monitor_points().to(self.device)  # (M, 2)

        # 3. Concern: PFN "Floor" Estimation
        # We use the efficient Horizon Batching we discussed earlier
        # to get the PFN's belief at every time step t.
        with torch.no_grad():
            # logits shape: (B*T, M, Num_Classes)
            # We take the expected value or a low quantile as the "Floor"
            logits = self._get_pfn_predictions_batched(obs_traj, mon_x)

            # Convert PFN logits to expected Y values
            # floor_y shape: (B*T, M) -> (T, B, M)
            floor_y = self.pfn.decode_to_mean(logits).view(T, B, -1)

        # 4. Calculate Surrogate Volume
        # Vol_t = Integral over Monitor Set of max(0, True_Incumbent_t - PFN_Floor_t)
        ceilings = incumbents.unsqueeze(-1)  # (T, B, 1)

        # This gap shrinks if:
        # A) Incumbent drops (Ceiling falls)
        # B) PFN realizes a region is bad (Floor rises)
        gaps = torch.clamp(ceilings - floor_y, min=1e-7)
        volumes = gaps.mean(dim=-1)  # (T, B)

        # 5. Reward: Log-Volume Reduction - Penalty
        log_v = torch.log(volumes)
        vol_reduction = log_v[:-1] - log_v[1:]

        first_step = torch.zeros(1, B, device=self.device)
        return torch.cat([first_step, vol_reduction], dim=0) - self.step_penalty

    def _get_pfn_predictions_batched(self, obs_traj, mon_x):
        # Implementation of the Block-Diagonal or Triangular batching
        # from our previous discussion to get PFN output for all T.
        pass


# You are absolutely right. Running a full PFN inference on a grid of 100+ points for every time step and every batch
# element is a computational nightmare that will turn your training time from hours into weeks.To get that "
# dense flashlight" signal without the PFN overhead, we can use In-Context Dispersion or Curvature-based Proxies.
# Here are three ways to get a low-cost proxy for expected volume reduction, ranked from "Cheapest" to "Most Sophisticated.
# "1. The "Distance-to-Incumbent" Weighted Novelty (The Cheapest)Instead of asking a PFN what it thinks about a grid,
# we use a geometric heuristic: Reward points that are both far from previous points (Exploration) AND have a high p
# robability of being better than the incumbent based on the Transformer's internal state.In your Transformer optimizer,
# the hidden state $h_t$ already contains the model's "belief." We can attach a tiny Value Head that predicts the
# "Expected Improvement" (EI) of the current action $x_t$ without querying a grid.Logic:
# $R_t = \text{ValueHead}(h_t) - \eta$Why it's cheap: It’s just one extra linear layer on top of your existing
# transformer tokens. No extra forward passes, no grids.The catch: You must train this head using the Ground Truth
# Volume as a target (Distillation).2. Analytical Proxy: The "Voronoi" Volume ReductionIf you don't want to use the
# PFN at all for the reward, you can use the Geometry of the Search Space. A point's "value" in reducing volume is
# proportional to the size of its Voronoi cell—the area of the domain it is "responsible" for.Low-cost implementation:
# Take your new point $x_t$.Calculate the distance to its nearest neighbor $x_{near}$.Proxy
# Reward: $R_t = \|x_t - x_{near}\|^2 \times \text{ReLU}(y_{incumbent} - y_t)$.Why this works: It rewards you for
# finding a "large empty hole" (High distance) and even more if that hole results in a good $y$ value.3. The
# "Representative Subset" PFN (The Sweet Spot)Instead of a grid of 1000 points, we use a Sparse Anchor Set (e.g.,
# only 8-16 points) to track the volume.If you use the Triangular/Block-Diagonal Masking we discussed, the marginal
# cost of adding a few test tokens is actually quite low because of how modern GPUs handle sequence length.Python
# def cheap_surrogate_proxy(pfn, obs_traj, env, num_anchors=8):
#     # 1. Use a very small Sobol set as 'Anchors' for the whole domain
#     anchors = torch.quasirandom.SobolEngine(2).draw(num_anchors) # (8, 2)
#
#     # 2. Query PFN only on these 8 points
#     # Using the causal mask logic, this is ONE forward pass.
#     # The 'volume' is simply the average gap at these 8 spots.
#     logits = pfn(obs_traj, test_x=anchors)
#
#     # 3. Calculate Volume on this tiny subset
#     # Reduction in this sparse volume is a high-correlation proxy for
#     # the total volume reduction.
# My Recommendation: The "In-Context Value Head"Since you are already using a Transformer, the most efficient path
# is to make the Transformer tell you its own confidence.Add a second output head to your Transformer
# (alongside the policy head) that predicts the Log-Volume of Regret.During Training: Calculate the GroundTruthVolume
# (the "God View") on a dense grid. This is your Label. The Loss: Train the Transformer's value head to predict this
# label: $\mathcal{L} = \| \text{Head}(h_t) - \text{TrueVol}_t \|^2$. The Reward: Use the predicted volume change as
# the reward for the RL policy.