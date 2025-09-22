import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model

from src.loss.auc_alternatives import find_element_wise_optimal_trajectory
from src.utils.bar_distribution import BarDistribution


class TinyCausalTransformer(nn.Module):
    def __init__(
            self,
            d_in=3, d_model=64, n_heads=4, max_len=100, n_layers=4,
            logger=None,
            *args, **kwargs
    ):
        super().__init__()
        self.d_in = d_in

        self.get_model(d_in, d_model, n_heads, max_len, n_layers, *args, **kwargs)
        # Project input up to model dim

        # Todo make this configurable

        self.exploration = 'rnd'  # 'rnd', 'rnd-sorted', 'generate', 'mixed'

    def get_model(self, d_in, d_model, n_heads, max_len, n_layers):
        self.in_proj = nn.Linear(d_in, d_model)

        self.transformer = GPT2Model(
            GPT2Config(
                n_embd=d_model,
                n_layer=n_layers,
                n_head=n_heads,
                n_positions=max_len,
                n_ctx=max_len,
                vocab_size=1_000,  # dummy, not used
            )
        )

        # Project back down to d_in and squash
        self.out_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_in),
            nn.Sigmoid()
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x, past_key_values=None, use_cache=False):
        """
        x: (B, S, d_in)
        """
        h = self.in_proj(x)  # (B, S, d_model)
        outputs = self.transformer(
            inputs_embeds=h, past_key_values=past_key_values, use_cache=use_cache
        )
        hidden = outputs.last_hidden_state  # (B, S, d_model)
        out = self.out_proj(hidden)
        return out, outputs.past_key_values

    @torch.no_grad()
    def generate(self, env, initial_condition, B=16, T=24):
        self.eval()
        start_T = initial_condition.shape[1]
        X = torch.zeros(B, start_T + T, self.d_in, device=self.device)

        X[:, :start_T, :] = initial_condition

        past = None
        for t in range(0, T):
            if t == 0:
                inp = X[:, :start_T, :]  # all initial tokens
            else:
                # only ever feed the last token seed so far + past
                inp = X[:, start_T + t - 1:start_T + t, :]

            out, past = self.forward(inp, past_key_values=past, use_cache=True)
            next_step = out[:, -1, :]
            X[:, start_T + t, :-1] = next_step[:, :-1]  # only coordinates
            X[:, start_T + t, -1] = env.evaluate(next_step[:, :-1])  # ground truth y value
        self.train()
        return X

    @torch.no_grad()
    def explore(self, step, env, B=16, T=24):
        """
        Random, but y sorted trajectory
        :param env:
        :param initial_condition:
        :param B:
        :param T:
        :return:
        """
        # FIXME! at some point we will want to trancend the rnd sequences:
        # if step % 2 == 0:

        # avoid representational collapse from initial clustering in the center
        # based on gpt and linear initialization on 0.5
        initial_condition = env.sample_initial_condition(B=B)
        X = self.generate(
            env, B=B, T=T - 1,
            initial_condition=initial_condition
        )  # already has the correct env y value!
        if self.exploration == 'generate':
            return X

        # FIXME: these trajectories are still hard to learn from: they jump a lot between the
        #  valleys. Maybe do local gradient descent steps from random points instead?
        #  This would require that the env is differentiable.
        X_rnd = torch.rand(B, T, self.d_in, device=self.device)
        X_rnd[..., -1] = env.evaluate(X_rnd[..., :-1])

        if self.exploration == 'rnd-sorted':
            sorted_indices = torch.argsort(X_rnd[..., -1], dim=1, descending=True)

            # Gather sorted tensor along T dimension using sorted indices
            X_rnd = torch.gather(
                X_rnd[..., :], 1,
                sorted_indices.unsqueeze(-1).expand(-1, -1, X_rnd.size(-1))
            )

        if self.exploration.startswith('rnd'):
            return X_rnd

        # Augment the generated sequence with elementwise random better alternatives
        A = 1

        min_sequence, _, _ = find_element_wise_optimal_trajectory(
            X.unsqueeze(1).repeat(1, A, 1, 1),
            X[..., -1].unsqueeze(1).repeat(1, A, 1),
            X_rnd.unsqueeze(1).repeat(1, A, 1, 1),

        )
        # fixme: use batch dim for alternatives instead!
        if self.exploration == 'mixed':
            return min_sequence.squeeze(1)

        raise ValueError(f"Unknown exploration strategy {self.exploration}")

    @torch.no_grad()
    def act(self, X):
        """
        Given a batch of sequences, return the next action (x,y) for each sequence.
        X: (B, S, d_in)
        returns: (B, d_in)
        """
        out, _ = self.forward(X)
        next_step = out[:, -1, :]
        return next_step


if __name__ == '__main__':
    import math

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TinyCausalTransformer(d_in=4, d_model=64, n_heads=4, max_len=32).to(device)


    class Env:
        def __init__(self, device):
            self.device = device
            # Example parameters for sine waves
            self.a1, self.b1, self.phi1 = 0.5, 2, 0.1
            self.a2, self.b2, self.phi2 = 0.3, 3, 0.2
            self.m = 0.1

        def rotate(self, x, y):
            # Identity rotation for simplicity
            return x, y

        def evaluate(self, x, y):
            x = x.to(self.device) if isinstance(x, torch.Tensor) else torch.tensor(x,
                                                                                   device=self.device)
            y = y.to(self.device) if isinstance(y, torch.Tensor) else torch.tensor(y,
                                                                                   device=self.device)
            x_r, y_r = self.rotate(x, y)
            val = (self.a1 * torch.sin(self.b1 * 2 * math.pi * x_r + self.phi1)
                   + self.a2 * torch.sin(self.b2 * 2 * math.pi * y_r + self.phi2)
                   + self.m * ((x_r + y_r) / 2.0))
            return val.clamp(0, 1)


    env = Env(device)

    B, T = 4, 16  # small batch and sequence length for example
    x0, y0 = 0.1, 0.7

    xs, ys, zs = model.generate(env, B=B, T=T, x0=x0, y0=y0)

    print("Generated xs:", xs)
    print("Generated ys:", ys)
    print("Environment values zs:", zs)
