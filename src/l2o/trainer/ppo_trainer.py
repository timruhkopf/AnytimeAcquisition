from tqdm import tqdm

import torch

import torch.nn as nn
import torch.optim as optim


from src.l2o.trainer.callbacks.abstract import CallbackHandler


class ContinuousPPOTrainer:

    def __init__(self, policy, env, reward_model, buffer, device, callbacks=None, **kwargs):
        """

        :param policy: Some Transformer-based policy model
        :param env: a vectorized environment, that supports reset() and step(), but does not return rewards
        :param reward_model: Rewards are computed either at token or sequence level, and to allow flexibility as
         e.g. lookahead rewards, we decouple reward computation from the environment.
        :param config:
        """
        self.policy = policy.to(device)
        self.device = device
        self.env = env
        self.reward_manager = reward_model
        self.buffer = buffer
        self.buffer.to(device)

        # FIXME: pass the optimizer externally to allow for more flexibility (e.g. different learning rates for different parameter groups)
        # Consider a scheduler as well, but for now we can just use a fixed learning rate.
        self.optimizer = optim.AdamW(policy.parameters(), lr=kwargs.get("learning_rate", 1e-5), eps=1e-8)

        # Note: Small Learning Rate for Transformers
        # LLM-based policies are sensitive. We use 5e-6 to 1e-5.
        # Standard RL (3e-4) will usually cause the Transformer to diverge.
        # self.optimizer = optim.AdamW(model.parameters(), lr=1e-5, eps=1e-8)

        # TODO factor this into config
        self.clip_coef = kwargs.get("clip_coef", 0.2)
        self.ent_coef = kwargs.get("ent_coef", 0.01)
        self.vf_coef = kwargs.get("vf_coef", 0.5)
        self.max_grad_norm = kwargs.get("max_grad_norm", 0.5)
        self.ppo_steps = kwargs.get("ppo_epochs", 4)
        self.batch_size = kwargs.get("batch_size", 64)  # Rollouts per batch

        self.callbacks = callbacks or []
        # TODO callbackhandler will need to be ddp rank aware to avoid multiple logging
        self.callback_handler = CallbackHandler(self.callbacks, trainer=self)

        self.stop_training = False

    def state_dict(self):
        return {
            "model_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }


    def train(self, total_iterations):

        self.callback_handler.on_event('on_train_start')
        progress_bar = tqdm(range(total_iterations), desc="PPO Training")
        for epoch in progress_bar:
            self.epoch = epoch
            if self.stop_training:
                print("Training stopped early.")
                break
            loss = self.train_step()
            description = f"Epoch {epoch + 1} | Loss: {loss:.4f}"
            progress_bar.set_description(description)

        self.callback_handler.on_event('log_on_train_end')
        self.callback_handler.on_event('on_train_end')

    def collect_rollouts(self, env, buffer, total_episodes_to_collect):
        """Orchestrates collection and computes rewards on full sequences."""
        self.buffer.reset()
        num_envs = env.num_envs
        num_iterations = total_episodes_to_collect // num_envs

        all_last_values = torch.zeros(total_episodes_to_collect, device=self.device)

        for i in range(num_iterations):
            # 1. Run the vectorized interaction
            traj, seeds, last_vals = self.policy._run_vectorized_episode(env)

            # 2. Store block in buffer
            start_idx = i * num_envs
            end_idx = start_idx + num_envs

            buffer.store_batch(
                traj["obs"], traj["acts"], traj["logprobs"],
                traj["values"], traj["dones"], seeds
            )

            # Bootstrap final value for GAE
            # The Purpose: "Infinite Horizon" Bootstrapping In PPO, we use Generalized Advantage Estimation (GAE)
            # to figure out if an action was good. The value of an action depends on the rewards you get now
            # plus all the rewards you expect to get in the future.However, your environment has a max_steps limit.
            # When the environment stops at $t=100$:Was the agent in a great position to keep winning?Or was it
            # about to fail?If we just assume the future reward is $0$, we bias the model to think the episode "died."
            # all_last_values contains the Critic’s prediction of the Expected Future Value beyond the last step.
            # By adding this to our GAE calculation, we tell the model: "Even though the episode ended here, this is
            # how much more reward we probably would have gotten."
            # FIXME: make all_last values the final regret?
            all_last_values[start_idx:end_idx] = last_vals

        # 3. Holistic Reward Calculation
        # We wait until the buffer is full to process rewards in one big batch
        # This is much faster if your reward_manager is a neural network.
        all_rewards = self.reward_manager(buffer.obs)
        buffer.store_rewards(all_rewards)

        return all_last_values


    def train_step(self):
        # 1. Collect Data
        last_val = self.collect_rollouts(
            env=self.env,
            buffer=self.buffer,
            total_episodes_to_collect=self.batch_size * self.ppo_steps
        )
        self.callback_handler.on_event("on_rollout_end", last_val=last_val)

        # 2. Compute Advantages
        dataloader = self.buffer.get_loader(last_val, batch_size=self.batch_size)

        # 3. PPO Optimization Loop
        for ppo_step in range(self.ppo_steps):

            self.callback_handler.on_event("on_policy_epoch_start", epoch=self.epoch, ppo_step=ppo_step)

            losses = []
            for batch in dataloader:
                b_obs, b_acts, b_old_logprobs, b_returns, b_advs, b_old_values = batch

                # Note: Advantage Whitening (Normalization)
                # We normalize advantages at the mini-batch level. This ensures
                # that the gradient updates have a mean of 0, preventing the
                # Transformer from developing a global bias toward specific x-coordinates.
                b_advs = (b_advs - b_advs.mean()) / (b_advs.std() + 1e-8)

                # --- Forward Pass on Full Sequences ---
                # We pass the full sequence (Batch, Time, 3) to the model.
                # The model will re-compute the causal mask internally.
                # No KV-cache needed here (we want gradients for everything).

                actor_params, values, _ = self.policy(x_raw=b_obs)

                # Re-evaluate distribution
                dist = self.policy.get_action_distribution(actor_params)
                new_log_probs = dist.log_prob(b_acts).sum(-1)
                entropy = dist.entropy().sum(-1)

                # --- PPO Losses ---
                # Note: PPO Clipping (Surrogate Objective)
                # As per arXiv:2512.01374v3, clipping is the primary defense
                # against 'Policy Staleness'. It prevents a single high-reward
                # trajectory from pushing the Transformer parameters into
                # regions where it forgets how to process sequences.
                # 1. Ratio
                logratio = new_log_probs - b_old_logprobs
                ratio = logratio.exp()

                # 2. Policy Loss (Clipped)
                pg_loss1 = -b_advs * ratio
                pg_loss2 = -b_advs * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # 3. Value Loss (Clipped or Unclipped)
                # Standard practice: simple MSE against returns
                # Note: Value Function Clipping
                # We often clip the value function update as well, so that the
                # Critic doesn't change too much in a single step.
                v_loss_unclipped = (values.squeeze() - b_returns) ** 2
                v_clipped = b_old_values + torch.clamp(values.squeeze() - b_old_values, -0.2, 0.2)
                v_loss_clipped = (v_clipped - b_returns) ** 2
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                # v_loss = F.mse_loss(values.squeeze(), b_returns)

                # 4. Entropy Loss (Bonus)
                entropy_loss = -entropy.mean()

                # Total Loss
                loss = pg_loss + self.vf_coef * v_loss + self.ent_coef * entropy_loss
                losses.append(loss.item())

                self.optimizer.zero_grad()
                loss.backward()

                self.callback_handler.on_event("on_policy_clipping", epoch=self.epoch, policy_step=ppo_step)

                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

            losses = torch.tensor(losses).mean().item()
            self.callback_handler.on_event("on_policy_epoch_end", losses=losses)

            if self.stop_training:
                # can be accessed from the callbacks!
                break

        return loss.item()

class PathwisePPOTrainer(ContinuousPPOTrainer):
    def __init__(self, model, env, config, lambda_pathwise=0.1):
        super().__init__(model, env, config)
        self.lambda_pathwise = lambda_pathwise
        # Upgrade buffer to the Meta version
        self.buffer = MetaTrajectoryBuffer(env.num_envs, env.max_steps, config.input_dim, env.device)

    def collect_rollouts(self):
        # 1. Standard Collection
        last_val = super().collect_rollouts()

        # 2. Store Hidden Env Params (Critical for re-evaluation)
        # We capture the *current* state of the vectorized environment
        self.buffer.store_env_params(self.env.freq, self.env.phase)

        return last_val

    def train_step(self):
        last_val = self.collect_rollouts()
        dataloader = self.buffer.get_loader_with_params(last_val, batch_size=self.batch_size)

        loss_avg = 0.0

        for ppo_step in range(self.ppo_steps):
            for batch in dataloader:
                # Unpack (Note the extra freq/phase tensors)
                b_obs, b_acts, b_old_logprobs, b_returns, b_advs, b_old_values, b_freqs, b_phases = batch

                # Normalize Advantages
                b_advs = (b_advs - b_advs.mean()) / (b_advs.std() + 1e-8)

                # --- 1. Forward Pass (Transformer) ---
                actor_params, values, _ = self.model(x_raw=b_obs)

                # --- 2. Action Distribution ---
                dist = self.model.get_action_distribution(actor_params)

                # --- 3. PPO Loss Calculation (Standard) ---
                new_log_probs = dist.log_prob(b_acts).sum(-1)
                entropy = dist.entropy().sum(-1)
                ratio = (new_log_probs - b_old_logprobs).exp()

                pg_loss1 = -b_advs * ratio
                pg_loss2 = -b_advs * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = F.mse_loss(values.squeeze(), b_returns)

                # --- 4. The Pathwise "Dream" Loss (New Logic) ---
                # Sample NEW actions from the CURRENT policy with gradients
                # rsample() ensures differentiation flows back to actor_params
                dream_actions = dist.rsample()

                # Differentiable Evaluation
                # We ask: "If we took this dream action in the original environment state, what is Y?"
                # b_freqs, b_phases: (Batch, 2)
                # dream_actions: (Batch, Seq, 2)
                dream_y = self.env.functional_evaluate(dream_actions, b_freqs, b_phases)

                # Loss: We want to MINIMIZE y.
                pathwise_loss = dream_y.mean()

                # --- 5. Combined Optimization ---
                # We combine standard PPO stability with the "greedy" pathwise gradient
                total_loss = pg_loss + self.vf_coef * v_loss \
                             - self.ent_coef * entropy.mean() \
                             + self.lambda_pathwise * pathwise_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                loss_avg += total_loss.item()

        return loss_avg / (self.ppo_steps * len(dataloader))