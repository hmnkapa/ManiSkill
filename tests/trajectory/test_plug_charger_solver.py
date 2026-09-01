import numpy as np
import sapien

from mani_skill.examples.motionplanning.panda.solutions.plug_charger import (
    INSERTION_Z_BIAS,
    _desired_charger_pose,
)


def test_desired_charger_pose_applies_socket_frame_insertion_bias():
    assert INSERTION_Z_BIAS == 5.0e-4
    goal = sapien.Pose([0.2, -0.1, 0.3])

    target = _desired_charger_pose(goal, -0.05)

    np.testing.assert_allclose(target.p, [0.15, -0.1, 0.3005], atol=1e-8)
