"""
Check out vectorized environments
https://gemini.google.com/app/2dc4387e54052c30

i.e. vmap for a MLP archetype with different weight samples, and inputs. Notice, that normalizing the BNN outputs will
give us the absolute 0 in BNN surface space, but not guarantee, that the surface has this point
"""
from typing import Dict

import torch
import torch.nn as nn
from torch.func import functional_call, stack_module_state, vmap
import copy


class BatchedTaskFamily:
    def __init__(self, num_inputs, num_outputs, batch_size=64, device="cuda"):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.batch_size = batch_size
        self.device = device

    def sample_batched_tasks(self, num_layers=8, num_hidden=64):
        """
        Creates B independent weight initializations for a single architecture backbone.
        Returns the reference model and a stacked parameter dictionary.
        """
        # 1. Instantiate B independent models with identical architecture
        models = [
            self._create_single_mlp(num_layers, num_hidden).to(self.device)
            for _ in range(self.batch_size)
        ]

        # 2. Stack their parameters and buffers into tensors of shape (B, ...)
        params, buffers = stack_module_state(models)
        base_model = copy.deepcopy(models[0])

        return base_model, params, buffers

    def evaluate(self, base_model, params, buffers, X: torch.Tensor) -> torch.Tensor:
        """
        Vectorized oracle query.
        X shape: (B, num_inputs) -> Returns y shape: (B, num_outputs)
        """

        def call_single(p, b, x_i):
            return functional_call(base_model, (p, b), (x_i.unsqueeze(0),)).squeeze(0)

        # vmap over batch dimension 0 for params, buffers, and inputs X
        return vmap(call_single)(params, buffers, X)

    def _create_single_mlp(self, num_layers, num_hidden):
        # Build your standard MLP here without preactivation/output noise for deterministic MCTS
        layers = [nn.Linear(self.num_inputs, num_hidden), nn.Tanh()]
        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(num_hidden, num_hidden), nn.Tanh()])
        layers.append(nn.Linear(num_hidden, self.num_outputs))

        model = nn.Sequential(*layers)
        model.eval()
        return model


from dataclasses import dataclass


@dataclass
class VectorizedState:
    X_buffer: torch.Tensor  # Shape: (B, T_max, num_inputs)
    y_buffer: torch.Tensor  # Shape: (B, T_max, num_outputs)
    incumbent: torch.Tensor  # Shape: (B,)
    step_idx: int  # Current time step t (0 to T_max - 1)
    max_budget: int  # Total steps T

    @property
    def remaining_budget(self) -> int:
        return self.max_budget - self.step_idx - 1

    def get_history(self):
        """Returns valid history up to current step without copying."""
        return (
            self.X_buffer[:, :self.step_idx + 1, :],
            self.y_buffer[:, :self.step_idx + 1, :]
        )

    def clone(self):
        """Zero-cost copy for branching B parallel MCTS trees."""
        return VectorizedState(
            X_buffer=self.X_buffer.clone(),
            y_buffer=self.y_buffer.clone(),
            incumbent=self.incumbent.clone(),
            step_idx=self.step_idx,
            max_budget=self.max_budget
        )


class VectorizedEnvironment:
    def __init__(self, task_family: BatchedTaskFamily, baseline_y0: float = 0.0):
        self.task_family = task_family
        self.baseline_y0 = baseline_y0

    def reset(self, max_budget: int, batch_size: int, device="cuda") -> VectorizedState:
        return VectorizedState(
            X_buffer=torch.empty((batch_size, max_budget, self.task_family.num_inputs), device=device),
            y_buffer=torch.empty((batch_size, max_budget, self.task_family.num_outputs), device=device),
            # Initialize incumbent to the fixed reference baseline y_0
            incumbent=torch.full((batch_size,), self.baseline_y0, device=device, dtype=torch.float32),
            step_idx=0,
            max_budget=max_budget
        )

    def step(
            self,
            state: VectorizedState,
            base_model: nn.Module,
            params: dict,
            buffers: dict,
            x: torch.Tensor
    ):
        """
        Executes one batched step across B environments simultaneously.
        x shape: (B, num_inputs)
        """
        # 1. Vectorized Oracle Query -> shape: (B, num_outputs)
        y = self.task_family.evaluate(base_model, params, buffers, x)

        # 2. Write to pre-allocated buffers in-place (no torch.cat memory overhead)
        state.X_buffer[:, state.step_idx, :] = x
        state.y_buffer[:, state.step_idx, :] = y

        # 3. Tensorized Reward & Incumbent Update (no .item() calls)
        # Assuming single-objective optimization on output index 0
        current_y_val = y[:, 0]

        # New incumbent is element-wise maximum across the batch
        new_incumbent = torch.maximum(state.incumbent, current_y_val)

        # Calculate improvement over previous incumbent
        delta_incumbent = new_incumbent - state.incumbent

        # FIXME: should we really do this? ONLY the duration the observation really was incumbent!
        # Batched AUC reward: delta * remaining_budget
        r_t = delta_incumbent * state.remaining_budget

        # 4. Advance state
        next_state = VectorizedState(
            X_buffer=state.X_buffer,
            y_buffer=state.y_buffer,
            incumbent=new_incumbent,
            step_idx=state.step_idx + 1,
            max_budget=state.max_budget
        )

        is_terminal = (next_state.step_idx >= state.max_budget)

        return next_state, r_t, is_terminal


if __name__ == '__main__':

    class Policy(nn.Module):
        def __init__(self, num_inputs, num_outputs):
            super().__init__()
            #
            self.model = nn.Sequential(
                nn.Linear(num_inputs, 32),
                nn.ReLU(),
                nn.Linear(32, num_outputs)
            )

        def forward(self, trajectory):
            x = trajectory[0][:, -1, :]  # fixme trajectory goes into the pfn
            if torch.isnan(x).any(): # this can happen initially and because of the
                x = torch.nan_to_num(x)
            return self.model(x)


    NUM_INPUTS = 3
    NUM_OUTPUTS = 1
    BATCH_SIZE = 64
    batched_tasks = BatchedTaskFamily(
        num_inputs=NUM_INPUTS,
        num_outputs=NUM_OUTPUTS,
        batch_size=BATCH_SIZE,
        device="cpu"
    )

    NUM_LAYERS = torch.randint(1, 10, (1,)).item()
    NUM_HIDDEN = torch.randint(10, 64, (1,)).item()

    base_model, params, buffers = batched_tasks.sample_batched_tasks(
        num_layers=NUM_LAYERS,
        num_hidden=NUM_HIDDEN
    )

    # X = torch.randn(BATCH_SIZE, NUM_INPUTS)
    # evaluated = batched_tasks.evaluate(base_model, params, buffers, X)
    # print(evaluated)

    policy = Policy(NUM_INPUTS, NUM_INPUTS)
    venv = VectorizedEnvironment(batched_tasks, baseline_y0=0.0)


    NUM_EPISODES = 100
    for episode in range(NUM_EPISODES):
        MAX_BUDGET = torch.randint(0, 20, (1,)).item()
        NUM_LAYERS = torch.randint(1, 10, (1,)).item()
        NUM_HIDDEN = torch.randint(10, 64, (1,)).item()
        base_model, params, buffers = batched_tasks.sample_batched_tasks(num_layers=NUM_LAYERS, num_hidden=NUM_HIDDEN)

        # state reset
        next_state = venv.reset(MAX_BUDGET, BATCH_SIZE, device="cpu")

        is_terminal=False

        while not is_terminal:



            X = policy(next_state.get_history())

            next_state, r_t, is_terminal = venv.step(next_state, base_model, params, buffers, X)





