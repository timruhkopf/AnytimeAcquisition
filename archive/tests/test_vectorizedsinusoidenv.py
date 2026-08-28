import pytest
import torch

from src.l2o.environment.sinusoid import VectorizedSinusoidEnv


@pytest.fixture
def env_config():
    return {
        "num_envs": 4,
        "max_steps": 10,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }


def test_seed_reproducibility(env_config):
    """Ensures that passing the same seeds results in identical env parameters."""
    env = VectorizedSinusoidEnv(**env_config)

    # 1. First reset with specific seeds
    initial_seeds = torch.randint(0, 2 ** 31, (env_config["num_envs"],))
    obs1, seeds1 = env.reset(seeds=initial_seeds)
    freq1, phase1 = env.freq.clone(), env.phase.clone()

    # 2. Reset with different seeds to 'pollute' the state
    obs_rnd, seed_rnd = env.reset()


    # 3. Reset back with the initial seeds
    obs2, seeds2 = env.reset(seeds=initial_seeds)
    freq2, phase2 = env.freq.clone(), env.phase.clone()

    # Assertions
    assert torch.equal(seeds1, seeds2), "Seeds stored in env should match input seeds"
    assert torch.allclose(freq1, freq2), "Frequencies did not match after restoration"
    assert torch.allclose(phase1, phase2), "Phases did not match after restoration"
    assert not torch.equal(seeds1, seed_rnd)
    # Note: obs contains x (randomly sampled).
    # Current reset() samples x via torch.rand, which depends on the GLOBAL seed.
    # If obs1 != obs2, it's because torch.rand isn't tied to the env-seed.


def test_parameter_isolation(env_config):
    """Ensures each environment in the vector gets unique params based on its own seed."""
    env = VectorizedSinusoidEnv(**env_config)

    # Use different seeds for each env in the batch
    seeds = torch.tensor([100, 200, 300, 400])
    env.reset(seeds=seeds)

    # Check that env 0 and env 1 don't have the same parameters
    assert not torch.allclose(env.freq[0], env.freq[1]), "Env params should be unique per seed"


def test_deterministic_evaluation(env_config):
    """Ensures evaluate() is a pure function of x and the seeded params."""
    env = VectorizedSinusoidEnv(**env_config)
    env.reset(seeds=torch.tensor([42] * env_config["num_envs"]))

    x = torch.rand(env_config["num_envs"], 2, device=env_config["device"])
    out1 = env.evaluate(x)
    out2 = env.evaluate(x)

    assert torch.allclose(out1, out2), "Evaluation must be deterministic for fixed x"