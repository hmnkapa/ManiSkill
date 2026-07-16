import pytest
import torch

from mani_skill.utils.skill_annotation import (
    SKILL_VOCAB,
    SkillAnnotationContext,
    get_annotation_bundle,
    get_annotation_bundle_for_env,
    id_to_skill,
    normalize_skill_context,
    project_3d_to_2d,
    project_pose_to_grasp_annotation_2d,
    skill_to_id,
)


def test_skill_vocab_and_schema_normalization():
    assert SKILL_VOCAB == ("none", "pick", "place", "insert", "screw", "push")
    for idx, skill in enumerate(SKILL_VOCAB):
        assert skill_to_id(skill) == idx
        assert id_to_skill(idx) == skill

    with pytest.raises(ValueError):
        skill_to_id("invalid")

    context = SkillAnnotationContext(
        skill=["pick", "place"],
        skill_state=["reach", "release"],
        phase=["grasp", "place"],
        target_point_world=torch.tensor([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]]),
        target_gripper_width=torch.tensor([0.04, 0.06]),
        active_object=["cube", "cube"],
        target_object=["cube", "goal"],
    )
    normalized = normalize_skill_context(context)

    assert normalized.skill == ["pick", "place"]
    assert normalized.skill_id.tolist() == [skill_to_id("pick"), skill_to_id("place")]
    assert normalized.target_point_world.shape == (2, 3)
    assert normalized.target_point_valid.tolist() == [True, True]
    assert normalized.target_pose_world.shape == (2, 4, 4)
    assert normalized.target_pose_valid.tolist() == [False, False]
    assert normalized.target_gripper_width_valid.tolist() == [True, True]

    raw_pose_context = SkillAnnotationContext(
        skill=["pick", "place"],
        target_pose_world=torch.tensor(
            [
                [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                [0.1, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            ]
        ),
    )
    raw_pose_normalized = normalize_skill_context(raw_pose_context)
    assert raw_pose_normalized.target_pose_world.shape == (2, 4, 4)
    assert raw_pose_normalized.target_pose_valid.tolist() == [True, True]


def test_projection_visible_and_invisible_points():
    intrinsic = torch.tensor(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]
    )
    extrinsic = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
    )

    projected = project_3d_to_2d(
        torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [1.0, 0.0, 1.0]]),
        intrinsic,
        extrinsic,
        (100, 100),
    )

    assert projected["point_uv"].shape == (3, 2)
    assert torch.allclose(projected["point_uv"][0], torch.tensor([50.0, 50.0]))
    assert projected["point_visible"].tolist() == [True, False, False]
    assert torch.all(projected["point_uv"][1:] == -1)

    batched = project_3d_to_2d(
        torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
        intrinsic[None].repeat(2, 1, 1),
        extrinsic[None].repeat(2, 1, 1),
        (100, 100),
    )
    assert batched["point_uv"].shape == (2, 2)
    assert batched["point_visible"].tolist() == [True, True]


def test_grasp_rectangle_projection_shape_and_visibility():
    intrinsic = torch.tensor(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]
    )
    extrinsic = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
    )
    pose = torch.eye(4)[None]
    pose[:, 2, 3] = 1.0

    projected = project_pose_to_grasp_annotation_2d(
        pose,
        intrinsic,
        extrinsic,
        (100, 100),
        gripper_width=torch.tensor([0.2]),
        grasp_height=0.2,
    )

    assert projected["grasp_rect_uv"].shape == (1, 4, 2)
    assert projected["grasp_visible"].tolist() == [True]
    expected = torch.tensor([[[40.0, 40.0], [60.0, 40.0], [60.0, 60.0], [40.0, 60.0]]])
    assert torch.allclose(projected["grasp_rect_uv"], expected)


def test_manager_previous_cache_falls_back_when_target_is_missing():
    valid_context = SkillAnnotationContext(
        skill="pick",
        target_point_world=torch.tensor([[0.0, 0.0, 1.0]]),
        target_pose_world=torch.eye(4)[None],
        target_gripper_width=torch.tensor([0.04]),
        active_object="cube",
        target_object="cube",
    )
    missing_target_context = SkillAnnotationContext(skill="pick")
    env = _ContextSequenceEnv([valid_context, missing_target_context, missing_target_context])

    first = get_annotation_bundle(env)
    assert first["skill"] == ["pick"]
    assert first["target"]["point_valid"].tolist() == [True]

    fallback = get_annotation_bundle(env)
    assert fallback["skill"] == ["pick"]
    assert fallback["debug"]["used_previous"].tolist() == [True]
    assert torch.allclose(fallback["target"]["point_world"], first["target"]["point_world"])

    no_fallback = get_annotation_bundle(env, use_previous=False)
    assert no_fallback["skill"] == ["none"]
    assert no_fallback["target"]["point_valid"].tolist() == [False]


def test_pick_cube_style_context_shapes_for_grasp_and_not_grasp():
    env = _PickCubeStyleProviderEnv()

    bundle = get_annotation_bundle(env)
    assert bundle["skill"] == ["pick", "place"]
    assert bundle["target"]["point_world"].shape == (2, 3)
    assert bundle["target"]["pose_world"].shape == (2, 4, 4)
    assert bundle["target"]["gripper_width"].shape == (2,)
    assert bundle["target"]["point_valid"].tolist() == [True, True]
    assert bundle["active_object"] == ["cube", "cube"]
    assert bundle["target_object"] == ["cube", "goal"]

    single = get_annotation_bundle_for_env(env, torch.tensor([1]))
    assert single["skill"] == ["place"]
    assert single["target"]["point_world"].shape == (1, 3)
    assert single["target"]["pose_world"].shape == (1, 4, 4)


class _ContextSequenceEnv:
    num_envs = 1
    device = "cpu"

    def __init__(self, contexts):
        self._contexts = list(contexts)
        self._call_idx = 0

    def get_skill_annotation_context(self, env_idx=None):
        context = self._contexts[min(self._call_idx, len(self._contexts) - 1)]
        self._call_idx += 1
        return context


class _PickCubeStyleProviderEnv:
    num_envs = 2
    device = "cpu"

    def __init__(self):
        self.is_grasped = torch.tensor([False, True])
        self.cube_pos = torch.tensor([[0.0, 0.0, 0.05], [0.1, 0.0, 0.05]])
        self.goal_pos = torch.tensor([[0.0, 0.0, 0.3], [0.2, 0.0, 0.3]])

    def get_skill_annotation_context(self, env_idx=None):
        if env_idx is None:
            indices = torch.arange(self.num_envs)
        elif torch.is_tensor(env_idx):
            indices = env_idx.flatten().long()
        else:
            indices = torch.tensor([env_idx], dtype=torch.long)

        is_grasped = self.is_grasped[indices]
        target_point = torch.where(
            is_grasped[:, None], self.goal_pos[indices], self.cube_pos[indices]
        )
        target_pose = torch.eye(4)[None].repeat(len(indices), 1, 1)
        target_pose[:, :3, 3] = target_point
        skills = ["place" if bool(item) else "pick" for item in is_grasped]
        phases = ["transport" if bool(item) else "approach" for item in is_grasped]

        return SkillAnnotationContext(
            skill=skills,
            skill_state=phases,
            phase=phases,
            target_point_world=target_point,
            target_pose_world=target_pose,
            target_gripper_width=torch.where(
                is_grasped,
                torch.full((len(indices),), 0.08),
                torch.full((len(indices),), 0.04),
            ),
            active_object=["cube"] * len(indices),
            target_object=["goal" if bool(item) else "cube" for item in is_grasped],
            task_meta={"task": "PickCube-style"},
        )
