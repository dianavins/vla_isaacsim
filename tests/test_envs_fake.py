import numpy as np

from edgevla.envs.base import EnvAdapter
from edgevla.envs.fake import FakeEnv


def test_fake_env_is_an_env_adapter():
    env = FakeEnv(suites=["spatial"], steps_per_episode=2, success_pattern=[True])
    assert isinstance(env, EnvAdapter)
    assert env.suites == ["spatial"]


def test_episode_ends_after_steps_and_reports_scripted_success():
    env = FakeEnv(
        suites=["spatial"],
        steps_per_episode=2,
        success_pattern=[True, False],
    )
    # Episode 1 -> success True
    env.reset()
    _, _, done1, _ = env.step(np.zeros(7))
    assert done1 is False
    _, _, done2, _ = env.step(np.zeros(7))
    assert done2 is True
    assert env.is_success() is True
    # Episode 2 -> success False (pattern advances)
    env.reset()
    env.step(np.zeros(7))
    env.step(np.zeros(7))
    assert env.is_success() is False
