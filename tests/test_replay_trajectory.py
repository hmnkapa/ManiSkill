import os
from types import SimpleNamespace

import numpy as np
import pytest

from mani_skill import ASSET_DIR
from mani_skill.trajectory.replay_trajectory import (
    replay_cpu_sim,
    replay_parallelized_sim,
)


class _FakeProgressBar:
    def reset(self, **kwargs):
        pass

    def set_description(self, description):
        pass

    def update(self, n=1):
        pass


class _FakeBaseEnv:
    device = "cpu"

    def __init__(self, num_envs):
        self.num_envs = num_envs
        self.state = np.zeros((num_envs, 1), dtype=np.float32)

    def set_state_dict(self, state):
        self.state = np.asarray(state["value"], dtype=np.float32).reshape(
            self.num_envs, -1
        )

    def get_obs(self):
        return self.state.copy()

    def render_human(self):
        pass


class _FakeSkillAnnotationRecorder:
    def __init__(self):
        self.reset_states = []

    def reset(self, base_env):
        self.reset_states.append(base_env.state.copy())


class _FakeReplayEnv:
    def __init__(self, num_envs):
        self.num_envs = num_envs
        self.base_env = _FakeBaseEnv(num_envs)
        self._skill_annotation_recorder = _FakeSkillAnnotationRecorder()
        self.step_start_states = []
        self.flushed_env_indices = []
        self._reset_recording_buffers()

    def _reset_recording_buffers(self):
        reset_state = np.full((self.num_envs, 1), -1.0, dtype=np.float32)
        self.base_env.state = reset_state.copy()
        self._trajectory_buffer = SimpleNamespace(
            state={"value": reset_state[None].copy()},
            observation=reset_state[None].copy(),
        )

    def reset(self, **kwargs):
        self._reset_recording_buffers()
        self._skill_annotation_recorder.reset(self.base_env)
        return self.base_env.get_obs(), {}

    def step(self, action):
        self.step_start_states.append(self.base_env.state.copy())
        zeros = np.zeros((self.num_envs,), dtype=bool)
        return self.base_env.get_obs(), None, zeros, zeros, {"success": False}

    def flush_trajectory(self, env_idxs_to_flush=None):
        self.flushed_env_indices.append(env_idxs_to_flush)

    def flush_video(self, **kwargs):
        pass


def _replay_args(**overrides):
    values = dict(
        allow_failure=True,
        discard_timeout=False,
        max_retry=0,
        num_envs=1,
        save_traj=True,
        save_video=False,
        target_control_mode=None,
        traj_path="fake.h5",
        use_env_states=False,
        use_first_env_state=True,
        vis=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _episode(episode_id, seed):
    return {
        "episode_id": episode_id,
        "episode_seed": seed,
        "elapsed_steps": 1,
        "control_mode": "pd_joint_delta_pos",
        "reset_kwargs": {"seed": seed},
    }


def _trajectory(initial_state, next_state):
    return {
        "env_states": {
            "value": np.asarray([[initial_state], [next_state]], dtype=np.float32)
        },
        "actions": np.zeros((1, 1), dtype=np.float32),
    }


def test_cpu_replay_resyncs_recorded_initial_frame_from_source_state_zero():
    env = _FakeReplayEnv(num_envs=1)
    episodes = [_episode(episode_id=0, seed=123)]
    trajectories = {"traj_0": _trajectory(initial_state=10, next_state=20)}

    replay_cpu_sim(
        _replay_args(), env, None, _FakeProgressBar(), episodes, trajectories
    )

    np.testing.assert_array_equal(
        env._trajectory_buffer.state["value"][0, :, 0], [10]
    )
    np.testing.assert_array_equal(env._trajectory_buffer.observation[0, :, 0], [10])
    np.testing.assert_array_equal(
        env._skill_annotation_recorder.reset_states[-1], [[10]]
    )
    np.testing.assert_array_equal(env.step_start_states[0], [[10]])


def test_parallel_replay_resyncs_recorded_initial_annotations_for_all_envs():
    env = _FakeReplayEnv(num_envs=2)
    episodes = [_episode(episode_id=0, seed=123), _episode(episode_id=1, seed=456)]
    trajectories = {
        "traj_0": _trajectory(initial_state=10, next_state=20),
        "traj_1": _trajectory(initial_state=30, next_state=40),
    }

    replay_parallelized_sim(
        _replay_args(num_envs=2),
        env,
        _FakeProgressBar(),
        episodes,
        trajectories,
    )

    np.testing.assert_array_equal(
        env._trajectory_buffer.state["value"][0, :, 0], [10, 30]
    )
    np.testing.assert_array_equal(
        env._trajectory_buffer.observation[0, :, 0], [10, 30]
    )
    np.testing.assert_array_equal(
        env._skill_annotation_recorder.reset_states[-1], [[10], [30]]
    )
    np.testing.assert_array_equal(env.step_start_states[0], [[10], [30]])


@pytest.mark.parametrize(
    "control_mode",
    [
        "pd_joint_delta_pos",
        "pd_joint_target_delta_pos",
        "pd_joint_vel",
        "pd_ee_delta_pose",
    ],
)
def test_replay_trajectory(control_mode):
    env_id = "PickCube-v1"
    # from mani_skill.utils.download_demo import main as download_demo, parse_args as download_demo_parse_args
    # download_demo(download_demo_parse_args(args=[env_id]))
    from mani_skill.trajectory.replay_trajectory import main, parse_args

    main(
        parse_args(
            args=[
                "--traj-path",
                f"{os.path.expandvars('$HOME')}/.maniskill/demos/{env_id}/teleop/trajectory.h5",
                "--save-traj",
                "--target-control-mode",
                control_mode,
                "--count",
                "4",
            ]
        )
    )
