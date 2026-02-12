import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Beta
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

# Hugging Face Imports
from transformers import (
    GPT2Config, GPT2Model,
    LlamaConfig, LlamaModel,
    PreTrainedModel
)


# ==========================================
# 1. The Model: Reusing HF Backbones
# ==========================================


class L2OPolicy(nn.Module):
    """
    Wraps a HF Transformer backbone.
    Bypasses the Embedding layer to input continuous vectors directly.
    """

    def __init__(self, input_dim, d_model, n_layer, n_head, max_len, pe_type, clamp_beta=100.0):
        """

        :param input_dim: The dimensionality of the continuous input (e.g. 3 for (x1, x2, y))
        :param d_model: The hidden size of the Transformer. A small model (64-128) is usually sufficient for simple tasks.
        :param n_layer: The number of Transformer layers. 2-4 layers are often enough for simple optimization tasks.
        :param n_head: The number of attention heads. 2-4 heads are usually sufficient for small models.
        :param max_len: The maximum sequence length (T). This should match the environment's max steps.
        :param pe_type: The type of positional encoding to use. 'learned' for GPT-2 style learned absolute embeddings, 'rope' for Llama-style Rotary Positional Embeddings.
        """
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.n_layer = n_layer
        self.n_head = n_head
        self.max_len = max_len
        self.pe_type = pe_type
        self.clamp_beta = clamp_beta

        # 1. Input Projection (Continuous -> d_model)
        self.input_proj = nn.Linear(input_dim, d_model)

        # 2. Backbone Selection (Existing HF Logic)
        if pe_type == "learned":
            # GPT-2 uses Learned Absolute Positional Embeddings
            hf_config = GPT2Config(
                n_embd=d_model,
                n_layer=n_layer,
                n_head=n_head,
                n_positions=max_len,
                use_cache=True
            )
            self.backbone = GPT2Model(hf_config)
        elif pe_type == "rope":
            # Llama uses Rotary Positional Embeddings (RoPE)
            hf_config = LlamaConfig(
                hidden_size=d_model,
                num_hidden_layers=n_layer,
                num_attention_heads=n_head,
                max_position_embeddings=max_len,
                use_cache=True,
                intermediate_size=d_model * 4
            )
            self.backbone = LlamaModel(hf_config)
        else:
            raise ValueError(f"Unknown PE type: {pe_type}")

        # 3. Heads
        # Actor: Outputs 4 params (alpha_x, beta_x, alpha_y, beta_y)
        self.actor_head = nn.Linear(d_model, 4)
        # Critic: Outputs scalar value
        self.critic_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            # Critical Check: Why Tanh? In PPO, the Critic needs to predict the "Return" ($G_t$).
            # If your rewards are sparse (0 or 1), Tanh is fine.The Risk: If your returns are large
            # (e.g., a path-finding reward of +50.0), a Tanh hidden layer can sometimes lead to
            # "dead neurons" early in training if the weights aren't initialized perfectly for the
            # scale of the inputs.
            nn.Linear(d_model, 1)
        )

        # Note: Orthogonal Initialization
        # We initialize the heads with small weights to ensure that at T=0,
        # the model outputs an almost uniform distribution. This prevents
        # the model from starting with huge 'certainty' and exploding gradients.
        nn.init.orthogonal_(self.critic_head[-1].weight, gain=1.0)

    def forward(self, inputs_embeds=None, past_key_values=None, x_raw=None):
        """
        Accepts either raw inputs (x_raw) to project, or pre-projected embeddings.
        """
        # 1. Project Continuous Inputs
        if inputs_embeds is None:
            assert x_raw is not None, "Must provide x_raw if inputs_embeds is None"
            inputs_embeds = self.input_proj(x_raw)

            # 2. Add Positional Embeddings (ONLY for Absolute PE / GPT-2)
            if isinstance(self.backbone, GPT2Model):
                # Notice, that GPT-2 in a forward would expect input_ids and would internally create position_ids
                # based on the sequence length. Since we use inputs_embeds directly, we bypass that internal logic
                # and need to create position_ids ourselves. This is not necessary for RoPE-based models like Llama,
                # which compute positional information dynamically in the attention mechanism itself!.
                batch_size, seq_length = x_raw.shape[:2]
                device = x_raw.device

                # Create position ids (0, 1, 2... T-1)
                position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
                position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_length)

                # Get embeddings from the backbone's internal storage
                position_embeds = self.backbone.wpe(position_ids)
                inputs_embeds = inputs_embeds + position_embeds

        # 2. Pass through Backbone
        # Note: We pass inputs_embeds directly, bypassing the backbone's internal discrete embedding layer
        outputs = self.backbone(
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=True
        )

        hidden_states = outputs.last_hidden_state
        new_past_key_values = outputs.past_key_values

        # 3. Heads
        # Softplus + 1.0 to ensure Alpha/Beta > 1.0 (Unimodal distribution)
        # This softplus is a duplicate to the one in get_action_distribution!
        # actor_params = F.softplus(self.actor_head(hidden_states)) + 1.0
        actor_params = self.actor_head(hidden_states)
        values = self.critic_head(hidden_states)

        return actor_params, values, new_past_key_values

    def get_action_distribution(self, actor_params):
        # Note: The Beta 'Concentration' Constraint
        # We output alpha and beta. We add +1.0 after softplus.
        # This ensures alpha, beta >= 1.0, keeping the distribution unimodal.
        # Without this, the agent might 'collapse' to the edges (0 or 1) too early.
        alpha = F.softplus(actor_params[..., :2]) + 1.01
        beta = F.softplus(actor_params[..., 2:]) + 1.01

        # CLAMPING: Prevent the distribution from becoming too 'sharp'
        # This prevents NaN gradients and 'frozen' policies
        alpha = torch.clamp(alpha, max=self.clamp_beta)
        beta = torch.clamp(beta, max=self.clamp_beta)

        return Beta(alpha, beta)

    @torch.no_grad()
    def _run_vectorized_episode(self, env):
        """Handles interaction with the vectorized environment  and KV-caching."""
        d = env.device  # fixme: to self.backbone.device?
        num_envs = env.num_envs
        max_steps = env.max_steps

        # Pre-allocate local tensors for this batch
        traj = {
            "obs": torch.zeros((num_envs, max_steps, 3), device=d),
            "acts": torch.zeros((num_envs, max_steps, 2), device=d),
            "logprobs": torch.zeros((num_envs, max_steps), device=d),
            "values": torch.zeros((num_envs, max_steps), device=d),
            "dones": torch.zeros((num_envs, max_steps), device=d),
        }

        obs, _ = env.reset()
        seeds = env.current_seeds.clone()
        curr_input = obs.unsqueeze(1)  # (N, 1, 3) for Transformer
        past_key_values = None

        for t in range(max_steps):
            # Policy Forward Pass
            actor_params, value, past_key_values = self.forward(
                x_raw=curr_input,
                past_key_values=past_key_values
            )

            dist = self.get_action_distribution(actor_params)
            action = dist.sample()  # (N, 1, 2)
            log_prob = dist.log_prob(action).sum(-1).squeeze(1)

            # Step Environment
            next_obs = env.step(action.squeeze(1))

            # Record data
            traj["obs"][:, t] = curr_input.squeeze(1)
            traj["acts"][:, t] = action.squeeze(1)
            traj["logprobs"][:, t] = log_prob
            traj["values"][:, t] = value.squeeze()
            traj["dones"][:, t] = float(t == max_steps - 1)

            curr_input = next_obs.unsqueeze(1)

        # Final bootstrap value for GAE
        _, last_val, _ = self.forward(x_raw=curr_input, past_key_values=past_key_values)

        return traj, seeds, last_val.squeeze()
