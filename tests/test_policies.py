import numpy as np

from edgevla.policies.base import VLAPolicy
from edgevla.policies.dummy import DummyPolicy


def test_dummy_policy_is_a_vla_policy():
    p = DummyPolicy(action_dim=7)
    assert isinstance(p, VLAPolicy)  # runtime-checkable Protocol
    assert p.name == "dummy"


def test_dummy_policy_acts_with_correct_shape():
    p = DummyPolicy(action_dim=7)
    p.reset()
    action = p.act(observation={"image": None})
    assert isinstance(action, np.ndarray)
    assert action.shape == (7,)
    assert np.all(action == 0.0)
