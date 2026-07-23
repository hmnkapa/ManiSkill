import numpy as np
import pytest
import torch

from mani_skill.envs.tasks.tabletop.lift_peg_upright import (
    LiftPegUprightSkillFSM,
    LiftPegUprightSkillPhase,
)
from mani_skill.envs.tasks.tabletop.peg_insertion_side import (
    PegInsertionSideSkillFSM,
    PegInsertionSideSkillPhase,
)
from mani_skill.envs.tasks.tabletop.pick_cube import (
    PickCubeSkillFSM,
    PickCubeSkillPhase,
)
from mani_skill.envs.tasks.tabletop.place_sphere import (
    PlaceSphereSkillFSM,
    PlaceSphereSkillPhase,
)
from mani_skill.envs.tasks.tabletop.poke_cube import (
    PokeCubeSkillFSM,
    PokeCubeSkillPhase,
)
from mani_skill.envs.tasks.tabletop.pull_cube import (
    PullCubeSkillFSM,
    PullCubeSkillPhase,
)
from mani_skill.envs.tasks.tabletop.pull_cube_tool import (
    PullCubeToolSkillFSM,
    PullCubeToolSkillPhase,
)
from mani_skill.envs.tasks.tabletop.plug_charger import (
    PlugChargerSkillFSM,
    PlugChargerSkillPhase,
)
from mani_skill.envs.tasks.tabletop.push_cube import (
    PushCubeSkillFSM,
    PushCubeSkillPhase,
)
from mani_skill.envs.tasks.tabletop.push_t import PushTSkillFSM, PushTSkillPhase
from mani_skill.envs.tasks.tabletop.roll_ball import (
    RollBallSkillFSM,
    RollBallSkillPhase,
)
from mani_skill.envs.tasks.tabletop.stack_cube import (
    StackCubeSkillFSM,
    StackCubeSkillPhase,
)
from mani_skill.envs.tasks.tabletop.stack_pyramid import (
    StackPyramidSkillFSM,
    StackPyramidSkillPhase,
)
from mani_skill.utils.skill_annotation import (
    SKILL_IDS,
    SKILL_NAMES,
    SKILL_VOCAB,
    SkillAnnotationEpisodeRecorder,
    SkillAnnotationContext,
    build_grasp_rect_corners_3d,
    get_annotation_bundle,
    get_annotation_bundle_for_env,
    id_to_skill,
    normalize_skill_context,
    project_3d_to_2d,
    project_pose_to_grasp_annotation_2d,
    reset_skill_annotator,
    skill_to_id,
)
from mani_skill.utils.structs.pose import Pose


def test_skill_vocab_and_schema_normalization():
    assert SKILL_IDS == {
        "none": 0,
        "pick": 1,
        "place": 2,
        "insert": 3,
        "screw": 4,
        "push": 5,
    }
    assert SKILL_NAMES == {
        0: "none",
        1: "pick",
        2: "place",
        3: "insert",
        4: "screw",
        5: "push",
    }
    assert SKILL_VOCAB == ("none", "pick", "place", "insert", "screw", "push")
    for idx, skill in enumerate(SKILL_VOCAB):
        assert skill_to_id(skill) == idx
        assert id_to_skill(idx) == skill

    with pytest.raises(ValueError):
        skill_to_id("invalid")

    context = SkillAnnotationContext(
        skill=["pick", "place"],
        skill_state=["reach", "release"],
        phase_id=torch.tensor([10, 20]),
        phase=["grasp", "place"],
        target_point_world=torch.tensor([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]]),
        target_gripper_width=torch.tensor([0.04, 0.06]),
        active_object=["cube", "cube"],
        target_object=["cube", "goal"],
    )
    normalized = normalize_skill_context(context)

    assert normalized.skill == ["pick", "place"]
    assert normalized.skill_id.tolist() == [skill_to_id("pick"), skill_to_id("place")]
    assert normalized.phase_id.tolist() == [10, 20]
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
    assert raw_pose_normalized.phase_id.tolist() == [-1, -1]


def test_bundle_skill_id_and_phase_id_semantics():
    skill_only_env = _ContextSequenceEnv(
        [
            SkillAnnotationContext(
                skill="pick",
                target_point_world=torch.tensor([[0.0, 0.0, 1.0]]),
            )
        ]
    )
    skill_only_bundle = get_annotation_bundle(skill_only_env)
    assert skill_only_bundle["skill"] == ["pick"]
    assert skill_only_bundle["skill_id"].shape == (1,)
    assert skill_only_bundle["skill_id"].tolist() == [SKILL_IDS["pick"]]
    assert skill_only_bundle["phase_id"].shape == (1,)
    assert skill_only_bundle["phase_id"].tolist() == [-1]

    explicit_id_env = _ContextSequenceEnv(
        [
            SkillAnnotationContext(
                skill="pick",
                skill_id=SKILL_IDS["place"],
                phase_id=7,
                phase="fsm-place",
                target_point_world=torch.tensor([[0.0, 0.0, 1.0]]),
            )
        ]
    )
    explicit_id_bundle = get_annotation_bundle(explicit_id_env)
    assert explicit_id_bundle["skill"] == ["place"]
    assert explicit_id_bundle["skill_id"].tolist() == [SKILL_IDS["place"]]
    assert explicit_id_bundle["phase_id"].tolist() == [7]
    assert explicit_id_bundle["phase"] == ["fsm-place"]


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
        grasp_height=0.1,
    )

    assert projected["grasp_rect_uv"].shape == (1, 4, 2)
    assert projected["grasp_visible"].tolist() == [True]
    expected = torch.tensor(
        [[[45.0, 40.0], [55.0, 40.0], [55.0, 60.0], [45.0, 60.0]]]
    )
    assert torch.allclose(projected["grasp_rect_uv"], expected)


def test_grasp_rectangle_uses_local_y_as_closing_axis():
    pose = torch.eye(4)[None]
    pose[:, :3, 3] = torch.tensor([1.0, 2.0, 3.0])

    corners = build_grasp_rect_corners_3d(
        pose,
        gripper_width=torch.tensor([0.08]),
        grasp_height=0.02,
    )

    expected = torch.tensor(
        [
            [
                [0.99, 1.96, 3.0],
                [1.01, 1.96, 3.0],
                [1.01, 2.04, 3.0],
                [0.99, 2.04, 3.0],
            ]
        ]
    )
    assert torch.allclose(corners, expected)


def test_manager_previous_cache_falls_back_when_target_is_missing():
    valid_context = SkillAnnotationContext(
        skill="pick",
        phase_id=42,
        target_point_world=torch.tensor([[0.0, 0.0, 1.0]]),
        target_pose_world=torch.eye(4)[None],
        target_gripper_width=torch.tensor([0.04]),
        active_object="cube",
        target_object="cube",
        task_meta={"task": "cache-test"},
    )
    missing_target_context = SkillAnnotationContext(skill="place", phase_id=99)
    env = _ContextSequenceEnv([valid_context, missing_target_context, missing_target_context])

    first = get_annotation_bundle(env)
    assert first["skill"] == ["pick"]
    assert first["skill_id"].tolist() == [SKILL_IDS["pick"]]
    assert first["phase_id"].tolist() == [42]
    assert first["target"]["point_valid"].tolist() == [True]
    assert first["task_meta"] == {"task": "cache-test"}

    fallback = get_annotation_bundle(env)
    assert fallback["skill"] == ["pick"]
    assert fallback["skill_id"].tolist() == [SKILL_IDS["pick"]]
    assert fallback["phase_id"].tolist() == [42]
    assert fallback["debug"]["used_previous"].tolist() == [True]
    assert torch.allclose(fallback["target"]["point_world"], first["target"]["point_world"])
    assert fallback["task_meta"] == {"task": "cache-test"}

    no_fallback = get_annotation_bundle(env, use_previous=False)
    assert no_fallback["skill"] == ["none"]
    assert no_fallback["skill_id"].tolist() == [SKILL_IDS["none"]]
    assert no_fallback["phase_id"].tolist() == [-1]
    assert no_fallback["target"]["point_valid"].tolist() == [False]


def test_manager_preserves_explicit_targetless_annotation():
    env = _ContextSequenceEnv(
        [
            SkillAnnotationContext(
                skill="push",
                phase_id=2,
                phase="coast",
                skill_state="wait_for_ball_to_reach_goal",
                active_object="ball",
                allow_no_target=True,
            )
        ]
    )

    bundle = get_annotation_bundle(env)

    assert bundle["skill"] == ["push"]
    assert bundle["skill_id"].tolist() == [SKILL_IDS["push"]]
    assert bundle["phase_id"].tolist() == [2]
    assert bundle["phase"] == ["coast"]
    assert bundle["skill_state"] == ["wait_for_ball_to_reach_goal"]
    assert bundle["target"]["point_valid"].tolist() == [False]
    assert bundle["target"]["pose_valid"].tolist() == [False]
    assert bundle["target"]["gripper_width_valid"].tolist() == [False]
    assert bundle["debug"]["current_valid"].tolist() == [True]
    assert bundle["debug"]["used_previous"].tolist() == [False]


def test_pick_cube_style_context_shapes_for_grasp_and_not_grasp():
    env = _PickCubeStyleProviderEnv()

    bundle = get_annotation_bundle(env)
    assert bundle["skill"] == ["pick", "place"]
    assert bundle["skill_id"].shape == (2,)
    assert bundle["skill_id"].tolist() == [SKILL_IDS["pick"], SKILL_IDS["place"]]
    assert bundle["phase_id"].shape == (2,)
    assert bundle["phase_id"].tolist() == [10, 20]
    assert bundle["target"]["point_world"].shape == (2, 3)
    assert bundle["target"]["pose_world"].shape == (2, 4, 4)
    assert bundle["target"]["gripper_width"].shape == (2,)
    assert bundle["target"]["point_valid"].tolist() == [True, True]
    assert bundle["active_object"] == ["cube", "cube"]
    assert bundle["target_object"] == ["cube", "goal"]
    assert bundle["task_meta"] == {"task": "PickCube-style"}

    for env_idx in (1, [1], (1,), np.array([1]), torch.tensor([1])):
        single = get_annotation_bundle_for_env(env, env_idx)
        assert single["skill"] == ["place"]
        assert single["skill_id"].shape == (1,)
        assert single["skill_id"].tolist() == [SKILL_IDS["place"]]
        assert single["phase_id"].shape == (1,)
        assert single["phase_id"].tolist() == [20]
        assert single["target"]["point_world"].shape == (1, 3)
        assert single["target"]["pose_world"].shape == (1, 4, 4)
        assert single["task_meta"] == {"task": "PickCube-style"}


def test_place_sphere_skill_fsm_transitions_through_release_and_settle():
    env = _PlaceSphereFSMEnv()
    fsm = PlaceSphereSkillFSM(num_envs=1, device="cpu")

    assert fsm.phase.tolist() == [int(PlaceSphereSkillPhase.PICK)]

    env.set_info(is_obj_grasped=True)
    fsm.update(env)
    assert fsm.phase.tolist() == [int(PlaceSphereSkillPhase.PLACE)]

    env.set_info(
        is_obj_grasped=False,
        is_obj_on_bin=False,
        is_obj_static=False,
        success=False,
    )
    fsm.update(env)
    assert fsm.phase.tolist() == [int(PlaceSphereSkillPhase.RELEASE)]

    fsm.update(env)
    assert fsm.phase.tolist() == [int(PlaceSphereSkillPhase.RELEASE)]

    env.set_info(is_obj_grasped=True)
    fsm.update(env)
    assert fsm.phase.tolist() == [int(PlaceSphereSkillPhase.PLACE)]

    env.set_info(
        is_obj_grasped=False,
        is_obj_on_bin=True,
        is_obj_static=False,
        success=False,
    )
    fsm.update(env)
    assert fsm.phase.tolist() == [int(PlaceSphereSkillPhase.RELEASE)]

    fsm.update(env)
    assert fsm.phase.tolist() == [int(PlaceSphereSkillPhase.RELEASE)]

    env.set_info(
        is_obj_grasped=False,
        is_obj_on_bin=False,
        is_obj_static=True,
        success=False,
    )
    fsm.update(env)
    assert fsm.phase.tolist() == [int(PlaceSphereSkillPhase.PICK)]

    env.set_info(is_obj_grasped=True)
    fsm.update(env)
    env.set_info(
        is_obj_grasped=False,
        is_obj_on_bin=True,
        is_obj_static=True,
        success=True,
    )
    fsm.update(env)
    assert fsm.phase.tolist() == [int(PlaceSphereSkillPhase.RELEASE)]

    fsm.update(env)
    assert fsm.phase.tolist() == [int(PlaceSphereSkillPhase.DONE)]

    env.set_info(
        is_obj_grasped=False,
        is_obj_on_bin=False,
        is_obj_static=True,
        success=False,
    )
    fsm.update(env)
    assert fsm.phase.tolist() == [int(PlaceSphereSkillPhase.DONE)]


def test_place_sphere_release_context_and_partial_update():
    env = _PlaceSphereFSMEnv(num_envs=2)
    fsm = PlaceSphereSkillFSM(num_envs=2, device="cpu")
    fsm.phase.copy_(
        torch.tensor(
            [
                int(PlaceSphereSkillPhase.PLACE),
                int(PlaceSphereSkillPhase.RELEASE),
            ],
            dtype=torch.long,
        )
    )

    context = fsm.build_context(env)
    normalized = normalize_skill_context(context, num_envs=2, device="cpu")
    assert context.skill == ["place", "place"]
    assert context.phase == ["place", "release"]
    assert context.skill_state == ["move_to_bin", "wait_for_sphere_to_settle"]
    assert normalized.skill_id.tolist() == [SKILL_IDS["place"], SKILL_IDS["place"]]
    assert normalized.phase_id.tolist() == [
        int(PlaceSphereSkillPhase.PLACE),
        int(PlaceSphereSkillPhase.RELEASE),
    ]
    assert normalized.target_pose_valid.tolist() == [True, True]
    assert normalized.target_point_valid.tolist() == [True, True]
    assert context.target_gripper_width is None
    assert context.active_object == ["sphere", "sphere"]
    assert normalized.target_object == ["tcp", "tcp"]
    assert torch.allclose(
        normalized.target_point_world,
        normalized.target_pose_world[:, :3, 3],
    )

    selected = fsm.build_context(env, env_idx=[1])
    assert selected.phase_id.tolist() == [int(PlaceSphereSkillPhase.RELEASE)]
    assert selected.phase == ["release"]
    assert selected.skill == ["place"]
    assert selected.skill_state == ["wait_for_sphere_to_settle"]
    assert selected.target_object == ["tcp"]
    selected_target_pose = normalize_skill_context(
        selected, num_envs=1, device="cpu"
    ).target_pose_world.clone()

    env.obj.set_pose(
        torch.tensor(
            [[0.0, 0.0, -0.01], [0.0, 0.0, -0.03]], dtype=torch.float32
        )
    )
    selected_after_fall = normalize_skill_context(
        fsm.build_context(env, env_idx=[1]), num_envs=1, device="cpu"
    )
    assert torch.allclose(selected_after_fall.target_pose_world, selected_target_pose)

    fsm.phase.fill_(int(PlaceSphereSkillPhase.PLACE))
    env.set_info(is_obj_grasped=[True, False])
    fsm.update(env, env_idx=[1])
    assert fsm.phase.tolist() == [
        int(PlaceSphereSkillPhase.PLACE),
        int(PlaceSphereSkillPhase.RELEASE),
    ]

    fsm.reset(env_idx=[1])
    assert fsm.phase.tolist() == [
        int(PlaceSphereSkillPhase.PLACE),
        int(PlaceSphereSkillPhase.PICK),
    ]
    assert fsm._cached_place_target_pose_valid.tolist() == [True, False]
    assert torch.isfinite(fsm._cached_place_target_pose_world[0]).all()
    assert torch.isnan(fsm._cached_place_target_pose_world[1]).all()


def test_place_sphere_release_target_is_frozen_until_regrasp():
    env = _PlaceSphereFSMEnv()
    fsm = PlaceSphereSkillFSM(num_envs=1, device="cpu")

    env.set_info(is_obj_grasped=True)
    fsm.update(env)
    place_context = normalize_skill_context(
        fsm.build_context(env), num_envs=1, device="cpu"
    )
    last_place_target = place_context.target_pose_world.clone()

    env.obj.set_pose(torch.tensor([[0.0, 0.0, -0.02]], dtype=torch.float32))
    env.set_info(is_obj_grasped=False, is_obj_static=False, success=False)
    fsm.update(env)
    release_context = normalize_skill_context(
        fsm.build_context(env), num_envs=1, device="cpu"
    )
    assert fsm.phase.tolist() == [int(PlaceSphereSkillPhase.RELEASE)]
    assert torch.allclose(release_context.target_pose_world, last_place_target)

    env.obj.set_pose(torch.tensor([[0.0, 0.0, -0.04]], dtype=torch.float32))
    falling_context = normalize_skill_context(
        fsm.build_context(env), num_envs=1, device="cpu"
    )
    assert torch.allclose(falling_context.target_pose_world, last_place_target)

    env.set_info(is_obj_grasped=True)
    fsm.update(env)
    regrasp_context = normalize_skill_context(
        fsm.build_context(env), num_envs=1, device="cpu"
    )
    assert fsm.phase.tolist() == [int(PlaceSphereSkillPhase.PLACE)]
    assert not torch.allclose(regrasp_context.target_pose_world, last_place_target)
    assert torch.allclose(
        regrasp_context.target_point_world,
        torch.tensor([[0.0, 0.0, 0.2925]], dtype=torch.float32),
        atol=1e-5,
    )


def test_stack_cube_skill_fsm_transitions_wait_for_cube_settle():
    env = _StackCubeFSMEnv()
    fsm = StackCubeSkillFSM(num_envs=1, device="cpu")

    assert fsm.phase.tolist() == [int(StackCubeSkillPhase.PICK)]

    env.set_info(is_cubeA_grasped=True)
    fsm.update(env)
    assert fsm.phase.tolist() == [int(StackCubeSkillPhase.PLACE)]

    env.set_info(is_cubeA_grasped=False, is_cubeA_on_cubeB=True, success=False)
    fsm.update(env)
    assert fsm.phase.tolist() == [int(StackCubeSkillPhase.PLACE)]

    env.set_info(is_cubeA_grasped=False, is_cubeA_on_cubeB=False, success=False)
    fsm.update(env)
    assert fsm.phase.tolist() == [int(StackCubeSkillPhase.PICK)]

    env.set_info(is_cubeA_grasped=True)
    fsm.update(env)
    env.set_info(is_cubeA_grasped=False, is_cubeA_on_cubeB=True, success=True)
    fsm.update(env)
    assert fsm.phase.tolist() == [int(StackCubeSkillPhase.DONE)]

    env.set_info(is_cubeA_grasped=False, is_cubeA_on_cubeB=False, success=False)
    fsm.update(env)
    assert fsm.phase.tolist() == [int(StackCubeSkillPhase.DONE)]


def test_poke_cube_skill_fsm_transitions_through_align_and_push():
    env = _PokeCubeFSMEnv()
    fsm = PokeCubeSkillFSM(num_envs=1, device="cpu")

    assert fsm.phase.tolist() == [int(PokeCubeSkillPhase.PICK)]

    env.set_info(is_peg_grasped=True)
    fsm.update(env)
    assert fsm.phase.tolist() == [int(PokeCubeSkillPhase.ALIGN)]
    context = fsm.build_context(env)
    assert context.skill_id.tolist() == [SKILL_IDS["push"]]
    assert context.phase == ["align"]
    assert context.target_object == ["tcp"]

    env.set_info(is_peg_grasped=True, is_peg_cube_fit=False)
    fsm.update(env)
    assert fsm.phase.tolist() == [int(PokeCubeSkillPhase.ALIGN)]

    env.set_info(is_peg_grasped=False, success=False)
    fsm.update(env)
    assert fsm.phase.tolist() == [int(PokeCubeSkillPhase.PICK)]

    env.set_info(is_peg_grasped=True)
    fsm.update(env)
    env.set_info(is_peg_grasped=True, is_peg_cube_fit=True)
    fsm.update(env)
    assert fsm.phase.tolist() == [int(PokeCubeSkillPhase.PUSH)]
    context = fsm.build_context(env)
    assert context.skill_id.tolist() == [SKILL_IDS["push"]]
    assert context.phase == ["push"]
    assert context.target_object == ["tcp"]

    env.set_info(is_peg_grasped=False, success=False)
    fsm.update(env)
    assert fsm.phase.tolist() == [int(PokeCubeSkillPhase.PICK)]

    env.set_info(is_peg_grasped=True)
    fsm.update(env)
    env.set_info(is_peg_grasped=True, is_peg_cube_fit=True)
    fsm.update(env)
    env.set_info(is_peg_grasped=True, success=True)
    fsm.update(env)
    assert fsm.phase.tolist() == [int(PokeCubeSkillPhase.DONE)]

    env.set_info(is_peg_grasped=False, success=False)
    fsm.update(env)
    assert fsm.phase.tolist() == [int(PokeCubeSkillPhase.DONE)]


def test_peg_insertion_side_skill_fsm_transitions_through_pre_insert():
    env = _PegInsertionSideFSMEnv()
    fsm = PegInsertionSideSkillFSM(num_envs=1, device="cpu")

    assert fsm.phase.tolist() == [int(PegInsertionSideSkillPhase.PICK)]

    env.set_info(is_grasped=True)
    fsm.update(env)
    assert fsm.phase.tolist() == [int(PegInsertionSideSkillPhase.PRE_INSERT)]
    assert env.agent.max_angle_calls[-1] == 20
    context = fsm.build_context(env)
    assert context.skill_id.tolist() == [SKILL_IDS["insert"]]
    assert context.phase == ["pre_insert"]
    assert context.target_object == ["tcp"]

    env.set_info(is_grasped=True, pre_inserted=False)
    fsm.update(env)
    assert fsm.phase.tolist() == [int(PegInsertionSideSkillPhase.PRE_INSERT)]

    env.set_info(is_grasped=False, success=False)
    fsm.update(env)
    assert fsm.phase.tolist() == [int(PegInsertionSideSkillPhase.PICK)]

    env.set_info(is_grasped=True)
    fsm.update(env)
    env.set_info(is_grasped=True, pre_inserted=True)
    fsm.update(env)
    assert fsm.phase.tolist() == [int(PegInsertionSideSkillPhase.INSERT)]
    context = fsm.build_context(env)
    assert context.skill_id.tolist() == [SKILL_IDS["insert"]]
    assert context.phase == ["insert"]
    assert context.target_object == ["tcp"]

    env.set_info(is_grasped=False, success=False)
    fsm.update(env)
    assert fsm.phase.tolist() == [int(PegInsertionSideSkillPhase.PICK)]

    env.set_info(is_grasped=True)
    fsm.update(env)
    env.set_info(is_grasped=True, pre_inserted=True)
    fsm.update(env)
    env.set_info(is_grasped=True, success=True)
    fsm.update(env)
    assert fsm.phase.tolist() == [int(PegInsertionSideSkillPhase.DONE)]

    env.set_info(is_grasped=False, success=False)
    fsm.update(env)
    assert fsm.phase.tolist() == [int(PegInsertionSideSkillPhase.DONE)]


def test_pull_cube_tool_skill_fsm_transitions_through_pull():
    env = _PullCubeToolFSMEnv()
    fsm = PullCubeToolSkillFSM(num_envs=1, device="cpu")

    assert fsm.phase.tolist() == [int(PullCubeToolSkillPhase.PICK)]

    env.set_tool_grasped(True)
    fsm.update(env)
    assert fsm.phase.tolist() == [int(PullCubeToolSkillPhase.ALIGN)]
    assert env.agent.max_angle_calls[-1] == 20
    context = fsm.build_context(env)
    assert context.skill_id.tolist() == [SKILL_IDS["push"]]
    assert context.phase == ["align"]
    assert context.skill_state == ["position_hook_behind_cube"]
    assert context.target_object == ["tcp"]
    assert torch.allclose(
        context.target_point_world,
        torch.tensor([[0.13, -0.067, 0.025]], dtype=torch.float32),
    )

    fsm.update(env)
    assert fsm.phase.tolist() == [int(PullCubeToolSkillPhase.ALIGN)]

    env.set_tool_positioned(True)
    fsm.update(env)
    assert fsm.phase.tolist() == [int(PullCubeToolSkillPhase.PULL)]
    context = fsm.build_context(env)
    assert context.skill_id.tolist() == [SKILL_IDS["push"]]
    assert context.phase == ["pull"]
    assert context.skill_state == ["pull_cube_into_workspace"]
    assert torch.allclose(
        context.target_point_world,
        torch.tensor([[-0.22, -0.067, 0.025]], dtype=torch.float32),
    )

    fixed_pull_target = context.target_pose_world.clone()
    env.cube.set_pose(env.cube.pose.p + torch.tensor([[0.1, 0.03, 0.0]]))
    env.agent.set_tcp_pos(env.agent.tcp.pose.p + torch.tensor([[-0.1, 0.0, 0.0]]))
    context = fsm.build_context(env)
    assert torch.allclose(context.target_pose_world, fixed_pull_target)

    env.set_info(success=True)
    fsm.update(env)
    assert fsm.phase.tolist() == [int(PullCubeToolSkillPhase.DONE)]

    env.set_tool_grasped(False)
    env.set_info(success=False)
    fsm.update(env)
    assert fsm.phase.tolist() == [int(PullCubeToolSkillPhase.DONE)]


def test_pull_cube_tool_skill_fsm_latches_pull_target_for_selected_envs():
    env = _PullCubeToolFSMEnv(num_envs=2)
    fsm = PullCubeToolSkillFSM(num_envs=2, device="cpu")

    env.set_tool_grasped([True, True])
    fsm.update(env)
    env.set_tool_positioned([False, True])
    fsm.update(env, env_idx=[1])

    assert fsm.phase.tolist() == [
        int(PullCubeToolSkillPhase.ALIGN),
        int(PullCubeToolSkillPhase.PULL),
    ]
    assert torch.isnan(fsm.pull_target_pos[0]).all()
    assert torch.allclose(
        fsm.pull_target_pos[1],
        torch.tensor([-0.22, -0.067, 0.025], dtype=torch.float32),
    )

    fixed_pull_target = fsm.build_context(env, env_idx=[1]).target_pose_world.clone()
    env.cube.set_pose(env.cube.pose.p + torch.tensor([[0.2, 0.0, 0.0]]))
    assert torch.allclose(
        fsm.build_context(env, env_idx=[1]).target_pose_world,
        fixed_pull_target,
    )

    fsm.reset(env_idx=[1])
    assert torch.isnan(fsm.pull_target_pos[1]).all()
    assert torch.isnan(fsm.pull_target_q[1]).all()


def test_push_cube_skill_fsm_transitions_through_push():
    env = _PushCubeFSMEnv()
    fsm = PushCubeSkillFSM(num_envs=1, device="cpu")

    assert fsm.phase.tolist() == [int(PushCubeSkillPhase.ALIGN)]

    fsm.update(env)
    assert fsm.phase.tolist() == [int(PushCubeSkillPhase.ALIGN)]

    env.move_tcp_to_push_pose()
    fsm.update(env)
    assert fsm.phase.tolist() == [int(PushCubeSkillPhase.PUSH)]
    context = fsm.build_context(env)
    assert context.skill_id.tolist() == [SKILL_IDS["push"]]
    assert context.phase == ["push"]
    assert context.skill_state == ["push_cube_to_goal"]
    assert context.target_object == ["tcp"]

    env.set_info(success=True)
    fsm.update(env)
    assert fsm.phase.tolist() == [int(PushCubeSkillPhase.DONE)]

    env.set_info(success=False)
    fsm.update(env)
    assert fsm.phase.tolist() == [int(PushCubeSkillPhase.DONE)]


def test_pull_cube_skill_fsm_transitions_through_pull():
    env = _PullCubeFSMEnv()
    fsm = PullCubeSkillFSM(num_envs=1, device="cpu")

    assert fsm.phase.tolist() == [int(PullCubeSkillPhase.ALIGN)]

    fsm.update(env)
    assert fsm.phase.tolist() == [int(PullCubeSkillPhase.ALIGN)]

    env.move_tcp_to_pull_pose()
    fsm.update(env)
    assert fsm.phase.tolist() == [int(PullCubeSkillPhase.PULL)]
    context = fsm.build_context(env)
    assert context.skill_id.tolist() == [SKILL_IDS["push"]]
    assert context.phase == ["pull"]
    assert context.skill_state == ["pull_cube_to_goal"]
    assert context.target_object == ["tcp"]

    env.set_info(success=True)
    fsm.update(env)
    assert fsm.phase.tolist() == [int(PullCubeSkillPhase.DONE)]

    env.set_info(success=False)
    fsm.update(env)
    assert fsm.phase.tolist() == [int(PullCubeSkillPhase.DONE)]


def test_roll_ball_skill_fsm_transitions_through_hit_and_coast():
    env = _RollBallFSMEnv()
    fsm = RollBallSkillFSM(num_envs=1, device="cpu")

    assert fsm.phase.tolist() == [int(RollBallSkillPhase.ALIGN)]

    fsm.update(env)
    assert fsm.phase.tolist() == [int(RollBallSkillPhase.ALIGN)]

    env.move_tcp_to_hit_pose()
    fsm.update(env)
    assert fsm.phase.tolist() == [int(RollBallSkillPhase.HIT)]
    context = fsm.build_context(env)
    assert context.skill_id.tolist() == [SKILL_IDS["push"]]
    assert context.phase == ["hit"]
    assert context.skill_state == ["hit_ball_toward_goal"]
    assert context.target_object == ["tcp"]

    env.reached_status.fill_(0.0)
    env.set_ball_velocity([0.0, -0.1, 0.0])
    fsm.update(env)
    assert fsm.phase.tolist() == [int(RollBallSkillPhase.HIT)]

    env.set_ball_velocity([0.0, 0.5, 0.0])
    fsm.update(env)
    assert fsm.phase.tolist() == [int(RollBallSkillPhase.HIT)]

    env.set_ball_velocity([0.0, -0.2, 0.0])
    fsm.update(env)
    assert fsm.phase.tolist() == [int(RollBallSkillPhase.COAST)]
    context = fsm.build_context(env)
    normalized = normalize_skill_context(context, num_envs=1, device="cpu")
    assert context.skill_id.tolist() == [SKILL_IDS["push"]]
    assert context.phase == ["coast"]
    assert context.skill_state == ["wait_for_ball_to_reach_goal"]
    assert context.target_object == [None]
    assert torch.isnan(context.target_pose_world).all()
    assert torch.isnan(context.target_point_world).all()
    assert torch.isnan(context.target_gripper_width).all()
    assert normalized.target_pose_valid.tolist() == [False]
    assert normalized.target_point_valid.tolist() == [False]
    assert normalized.target_gripper_width_valid.tolist() == [False]
    assert normalized.allow_no_target.tolist() == [True]

    fsm.update(env)
    assert fsm.phase.tolist() == [int(RollBallSkillPhase.COAST)]

    env.set_info(success=True)
    fsm.update(env)
    assert fsm.phase.tolist() == [int(RollBallSkillPhase.DONE)]

    env.set_info(success=False)
    fsm.update(env)
    assert fsm.phase.tolist() == [int(RollBallSkillPhase.DONE)]


def test_roll_ball_skill_fsm_partial_update_uses_selected_ball_velocity():
    env = _RollBallFSMEnv(num_envs=2)
    fsm = RollBallSkillFSM(num_envs=2, device="cpu")
    fsm.phase.fill_(int(RollBallSkillPhase.HIT))
    env.set_ball_velocity(
        [
            [0.0, 0.5, 0.0],
            [0.0, -0.2, 0.0],
        ]
    )

    fsm.update(env, env_idx=[1])

    assert fsm.phase.tolist() == [
        int(RollBallSkillPhase.HIT),
        int(RollBallSkillPhase.COAST),
    ]
    context = fsm.build_context(env, env_idx=[1])
    assert context.phase_id.tolist() == [int(RollBallSkillPhase.COAST)]
    assert context.task_meta["ball_forward_speed"].shape == (1,)
    assert context.task_meta["is_ball_rolling_toward_goal"].tolist() == [True]

    fsm.reset(env_idx=[1])
    assert fsm.phase.tolist() == [
        int(RollBallSkillPhase.HIT),
        int(RollBallSkillPhase.ALIGN),
    ]


def test_lift_peg_upright_skill_fsm_transitions_through_place_stages():
    env = _LiftPegUprightFSMEnv()
    fsm = LiftPegUprightSkillFSM(num_envs=1, device="cpu")

    assert fsm.phase.tolist() == [int(LiftPegUprightSkillPhase.PICK)]

    env.set_grasped(True)
    fsm.update(env)
    assert fsm.phase.tolist() == [int(LiftPegUprightSkillPhase.LIFT)]
    context = fsm.build_context(env)
    assert context.skill_id.tolist() == [SKILL_IDS["place"]]
    assert context.phase == ["lift"]
    assert context.skill_state == ["lift_peg_to_safe_height"]
    assert torch.allclose(context.task_meta["safe_peg_height"], torch.tensor([0.22]))
    assert torch.allclose(context.task_meta["lift_height_error"], torch.tensor([0.22]))
    assert context.task_meta["is_lifted"].tolist() == [False]

    fsm.update(env)
    assert fsm.phase.tolist() == [int(LiftPegUprightSkillPhase.LIFT)]

    env.peg.set_pose(
        torch.tensor([[0.0, 0.0, 0.22]], dtype=torch.float32),
        torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
    )
    fsm.update(env)
    assert fsm.phase.tolist() == [int(LiftPegUprightSkillPhase.ROTATE)]
    context = fsm.build_context(env)
    assert context.skill_id.tolist() == [SKILL_IDS["place"]]
    assert context.phase == ["rotate"]
    assert context.skill_state == ["rotate_peg_upright"]
    assert context.task_meta["is_lifted"].tolist() == [True]
    assert context.task_meta["orientation_aligned"].tolist() == [False]
    rotate_target_pose = context.target_pose_world.clone()
    assert torch.allclose(
        context.target_point_world, env.agent.tcp.pose.p, atol=1e-5
    )

    env.agent.set_tcp_pos(torch.tensor([[-0.08, 0.02, 0.21]]))
    env.peg.set_pose(
        torch.tensor([[0.01, 0.02, 0.21]], dtype=torch.float32),
        torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
    )
    fsm.update(env)
    assert fsm.phase.tolist() == [int(LiftPegUprightSkillPhase.ROTATE)]
    context = fsm.build_context(env)
    assert torch.allclose(context.target_pose_world, rotate_target_pose, atol=1e-5)

    env.peg.set_pose(
        torch.tensor([[0.0, 0.0, 0.22]], dtype=torch.float32),
        torch.tensor([[0.5, 0.5, -0.5, 0.5]], dtype=torch.float32),
    )
    fsm.update(env)
    assert fsm.phase.tolist() == [int(LiftPegUprightSkillPhase.LOWER)]
    context = fsm.build_context(env)
    assert context.skill_id.tolist() == [SKILL_IDS["place"]]
    assert context.phase == ["lower"]
    assert context.skill_state == ["lower_peg_to_table"]
    assert context.task_meta["orientation_aligned"].tolist() == [True]

    fsm.update(env)
    assert fsm.phase.tolist() == [int(LiftPegUprightSkillPhase.LOWER)]

    env.set_info(success=True)
    fsm.update(env)
    assert fsm.phase.tolist() == [int(LiftPegUprightSkillPhase.DONE)]

    env.set_info(success=False)
    fsm.update(env)
    assert fsm.phase.tolist() == [int(LiftPegUprightSkillPhase.DONE)]

    for active_phase in (
        LiftPegUprightSkillPhase.LIFT,
        LiftPegUprightSkillPhase.ROTATE,
        LiftPegUprightSkillPhase.LOWER,
    ):
        fsm.phase.fill_(int(active_phase))
        env.set_grasped(False)
        fsm.update(env)
        assert fsm.phase.tolist() == [int(LiftPegUprightSkillPhase.PICK)]

    env = _LiftPegUprightFSMEnv(num_envs=2)
    fsm = LiftPegUprightSkillFSM(num_envs=2, device="cpu")
    env.set_grasped([False, True])
    fsm.update(env, env_idx=[1])
    assert fsm.phase.tolist() == [
        int(LiftPegUprightSkillPhase.PICK),
        int(LiftPegUprightSkillPhase.LIFT),
    ]

    peg_pos = env.peg.pose.p.clone()
    peg_pos[1, 2] = 0.22
    peg_q = env.peg.pose.q.clone()
    peg_q[1] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    env.peg.set_pose(peg_pos, peg_q)
    fsm.update(env, env_idx=[1])
    assert fsm.phase.tolist() == [
        int(LiftPegUprightSkillPhase.PICK),
        int(LiftPegUprightSkillPhase.ROTATE),
    ]


def test_plug_charger_skill_fsm_transitions_through_insert():
    env = _PlugChargerFSMEnv()
    fsm = PlugChargerSkillFSM(num_envs=1, device="cpu")

    assert fsm.phase.tolist() == [int(PlugChargerSkillPhase.PICK)]

    env.set_grasped(True)
    fsm.update(env)
    assert fsm.phase.tolist() == [int(PlugChargerSkillPhase.PRE_INSERT)]
    assert env.agent.max_angle_calls[-1] == 20
    context = fsm.build_context(env)
    assert context.skill_id.tolist() == [SKILL_IDS["insert"]]
    assert context.phase == ["pre_insert"]
    assert context.skill_state == ["align_charger_with_receptacle"]

    fsm.update(env)
    assert fsm.phase.tolist() == [int(PlugChargerSkillPhase.PRE_INSERT)]

    env.move_charger_to_pre_insert_pose()
    fsm.update(env)
    assert fsm.phase.tolist() == [int(PlugChargerSkillPhase.INSERT)]
    context = fsm.build_context(env)
    assert context.skill_id.tolist() == [SKILL_IDS["insert"]]
    assert context.phase == ["insert"]
    assert context.skill_state == ["insert_charger"]

    env.set_info(success=True)
    fsm.update(env)
    assert fsm.phase.tolist() == [int(PlugChargerSkillPhase.DONE)]

    env.set_grasped(False)
    env.set_info(success=False)
    fsm.update(env)
    assert fsm.phase.tolist() == [int(PlugChargerSkillPhase.DONE)]

    env = _PlugChargerFSMEnv(num_envs=2)
    fsm = PlugChargerSkillFSM(num_envs=2, device="cpu")
    env.set_grasped([False, True])
    fsm.update(env, env_idx=[1])
    assert fsm.phase.tolist() == [
        int(PlugChargerSkillPhase.PICK),
        int(PlugChargerSkillPhase.PRE_INSERT),
    ]


def test_push_t_skill_fsm_transitions_through_place():
    env = _PushTFSMEnv()
    fsm = PushTSkillFSM(num_envs=1, device="cpu")

    assert fsm.phase.tolist() == [int(PushTSkillPhase.PUSH_ORIENT)]

    fsm.update(env)
    assert fsm.phase.tolist() == [int(PushTSkillPhase.PUSH_ORIENT)]

    env.set_tee_yaw(0.0)
    fsm.update(env)
    assert fsm.phase.tolist() == [int(PushTSkillPhase.PUSH_PLACE)]
    context = fsm.build_context(env)
    assert context.skill_id.tolist() == [SKILL_IDS["push"]]
    assert context.phase == ["push_place"]
    assert context.skill_state == ["push_aligned_tee_into_goal"]

    env.set_intersection(0.90)
    fsm.update(env)
    assert fsm.phase.tolist() == [int(PushTSkillPhase.DONE)]

    env.set_intersection(0.0)
    fsm.update(env)
    assert fsm.phase.tolist() == [int(PushTSkillPhase.DONE)]

    env = _PushTFSMEnv(num_envs=2, tee_yaw=[0.2, 0.0])
    fsm = PushTSkillFSM(num_envs=2, device="cpu")
    fsm.update(env, env_idx=[1])
    assert fsm.phase.tolist() == [
        int(PushTSkillPhase.PUSH_ORIENT),
        int(PushTSkillPhase.PUSH_PLACE),
    ]


def test_stack_pyramid_skill_fsm_transitions_through_top_place():
    env = _StackPyramidFSMEnv()
    fsm = StackPyramidSkillFSM(num_envs=1, device="cpu")

    assert fsm.phase.tolist() == [int(StackPyramidSkillPhase.PUSH_BASE)]

    fsm.update(env)
    assert fsm.phase.tolist() == [int(StackPyramidSkillPhase.PUSH_BASE)]

    env.move_base_pair_ready()
    fsm.update(env)
    assert fsm.phase.tolist() == [int(StackPyramidSkillPhase.PICK_TOP)]
    context = fsm.build_context(env)
    assert context.skill_id.tolist() == [SKILL_IDS["pick"]]
    assert context.phase == ["pick_top"]
    assert context.active_object == ["cubeC"]

    env.set_cubeC_grasped(True)
    fsm.update(env)
    assert fsm.phase.tolist() == [int(StackPyramidSkillPhase.PLACE_TOP)]
    context = fsm.build_context(env)
    assert context.skill_id.tolist() == [SKILL_IDS["place"]]
    assert context.phase == ["place_top"]
    assert context.skill_state == ["place_cubeC_on_base_pair"]

    env.set_info(success=True)
    fsm.update(env)
    assert fsm.phase.tolist() == [int(StackPyramidSkillPhase.DONE)]

    env.set_info(success=False)
    fsm.update(env)
    assert fsm.phase.tolist() == [int(StackPyramidSkillPhase.DONE)]

    env = _StackPyramidFSMEnv(num_envs=2)
    fsm = StackPyramidSkillFSM(num_envs=2, device="cpu")
    env.move_base_pair_ready()
    fsm.update(env, env_idx=[1])
    assert fsm.phase.tolist() == [
        int(StackPyramidSkillPhase.PUSH_BASE),
        int(StackPyramidSkillPhase.PICK_TOP),
    ]


def test_tabletop_skill_fsm_contexts_normalize_for_all_tcp_target_tasks():
    cases = [
        (
            PickCubeSkillFSM,
            _PickCubeFSMEnv(num_envs=2),
            [PickCubeSkillPhase.PICK, PickCubeSkillPhase.PLACE],
            [SKILL_IDS["pick"], SKILL_IDS["place"]],
            "PickCube-v1",
        ),
        (
            PlaceSphereSkillFSM,
            _PlaceSphereFSMEnv(num_envs=2),
            [PlaceSphereSkillPhase.PICK, PlaceSphereSkillPhase.PLACE],
            [SKILL_IDS["pick"], SKILL_IDS["place"]],
            "PlaceSphere-v1",
        ),
        (
            StackCubeSkillFSM,
            _StackCubeFSMEnv(num_envs=2),
            [StackCubeSkillPhase.PICK, StackCubeSkillPhase.PLACE],
            [SKILL_IDS["pick"], SKILL_IDS["place"]],
            "StackCube-v1",
        ),
        (
            LiftPegUprightSkillFSM,
            _LiftPegUprightFSMEnv(num_envs=2),
            [LiftPegUprightSkillPhase.PICK, LiftPegUprightSkillPhase.LIFT],
            [SKILL_IDS["pick"], SKILL_IDS["place"]],
            "LiftPegUpright-v1",
        ),
        (
            PokeCubeSkillFSM,
            _PokeCubeFSMEnv(num_envs=2),
            [PokeCubeSkillPhase.PICK, PokeCubeSkillPhase.ALIGN],
            [SKILL_IDS["pick"], SKILL_IDS["push"]],
            "PokeCube-v1",
        ),
        (
            PegInsertionSideSkillFSM,
            _PegInsertionSideFSMEnv(num_envs=2),
            [
                PegInsertionSideSkillPhase.PICK,
                PegInsertionSideSkillPhase.PRE_INSERT,
            ],
            [SKILL_IDS["pick"], SKILL_IDS["insert"]],
            "PegInsertionSide-v1",
        ),
        (
            PlugChargerSkillFSM,
            _PlugChargerFSMEnv(num_envs=2),
            [PlugChargerSkillPhase.PICK, PlugChargerSkillPhase.PRE_INSERT],
            [SKILL_IDS["pick"], SKILL_IDS["insert"]],
            "PlugCharger-v1",
        ),
        (
            PullCubeToolSkillFSM,
            _PullCubeToolFSMEnv(num_envs=2),
            [PullCubeToolSkillPhase.PICK, PullCubeToolSkillPhase.ALIGN],
            [SKILL_IDS["pick"], SKILL_IDS["push"]],
            "PullCubeTool-v1",
        ),
        (
            PushCubeSkillFSM,
            _PushCubeFSMEnv(num_envs=2),
            [PushCubeSkillPhase.ALIGN, PushCubeSkillPhase.PUSH],
            [SKILL_IDS["push"], SKILL_IDS["push"]],
            "PushCube-v1",
        ),
        (
            PullCubeSkillFSM,
            _PullCubeFSMEnv(num_envs=2),
            [PullCubeSkillPhase.ALIGN, PullCubeSkillPhase.PULL],
            [SKILL_IDS["push"], SKILL_IDS["push"]],
            "PullCube-v1",
        ),
        (
            RollBallSkillFSM,
            _RollBallFSMEnv(num_envs=2),
            [RollBallSkillPhase.ALIGN, RollBallSkillPhase.HIT],
            [SKILL_IDS["push"], SKILL_IDS["push"]],
            "RollBall-v1",
        ),
        (
            PushTSkillFSM,
            _PushTFSMEnv(num_envs=2),
            [PushTSkillPhase.PUSH_ORIENT, PushTSkillPhase.PUSH_PLACE],
            [SKILL_IDS["push"], SKILL_IDS["push"]],
            "PushT-v1",
        ),
        (
            StackPyramidSkillFSM,
            _StackPyramidFSMEnv(num_envs=2),
            [StackPyramidSkillPhase.PUSH_BASE, StackPyramidSkillPhase.PICK_TOP],
            [SKILL_IDS["push"], SKILL_IDS["pick"]],
            "StackPyramid-v1",
        ),
    ]

    for fsm_cls, env, phases, skill_ids, task_name in cases:
        fsm = fsm_cls(num_envs=2, device="cpu")
        fsm.phase.copy_(torch.tensor([int(x) for x in phases], dtype=torch.long))

        normalized = normalize_skill_context(
            fsm.build_context(env), num_envs=2, device="cpu"
        )

        assert normalized.skill_id.tolist() == skill_ids
        assert normalized.phase_id.tolist() == [0, 1]
        assert normalized.target_point_world.shape == (2, 3)
        assert normalized.target_point_valid.tolist() == [True, True]
        assert normalized.target_pose_valid.tolist() == [True, True]
        assert normalized.target_object == ["tcp", "tcp"]
        assert torch.allclose(
            normalized.target_point_world,
            normalized.target_pose_world[:, :3, 3],
        )
        assert normalized.task_meta["task"] == task_name
        assert normalized.task_meta["target_frame"] == "world"

        selected = normalize_skill_context(
            fsm.build_context(env, env_idx=[1]), num_envs=1, device="cpu"
        )
        assert selected.phase_id.tolist() == [1]
        assert selected.target_object == ["tcp"]
        assert torch.allclose(
            selected.target_pose_world,
            normalized.target_pose_world[1:2],
        )
        assert torch.allclose(
            selected.target_point_world,
            normalized.target_point_world[1:2],
        )


def test_tabletop_skill_fsm_tcp_target_geometry_for_every_active_phase():
    cases = [
        (
            PickCubeSkillFSM,
            _PickCubeFSMEnv(num_envs=2),
            [PickCubeSkillPhase.PICK, PickCubeSkillPhase.PLACE],
            [[0.0, 0.0, 0.0], [0.1, 0.0, 0.3]],
        ),
        (
            PlaceSphereSkillFSM,
            _PlaceSphereFSMEnv(num_envs=3),
            [
                PlaceSphereSkillPhase.PICK,
                PlaceSphereSkillPhase.PLACE,
                PlaceSphereSkillPhase.RELEASE,
            ],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.2525], [0.0, 0.0, 0.2525]],
        ),
        (
            StackCubeSkillFSM,
            _StackCubeFSMEnv(num_envs=2),
            [StackCubeSkillPhase.PICK, StackCubeSkillPhase.PLACE],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.26]],
        ),
        (
            PokeCubeSkillFSM,
            _PokeCubeFSMEnv(num_envs=3),
            [
                PokeCubeSkillPhase.PICK,
                PokeCubeSkillPhase.ALIGN,
                PokeCubeSkillPhase.PUSH,
            ],
            [[0.0, 0.0, 0.0], [-0.12, 0.0, 0.2], [0.0, 0.0, 0.2]],
        ),
        (
            PegInsertionSideSkillFSM,
            _PegInsertionSideFSMEnv(num_envs=3),
            [
                PegInsertionSideSkillPhase.PICK,
                PegInsertionSideSkillPhase.PRE_INSERT,
                PegInsertionSideSkillPhase.INSERT,
            ],
            [[-0.06, 0.0, 0.0], [-0.01, 0.0, 0.25], [0.15, 0.0, 0.25]],
        ),
        (
            PullCubeToolSkillFSM,
            _PullCubeToolFSMEnv(num_envs=3),
            [
                PullCubeToolSkillPhase.PICK,
                PullCubeToolSkillPhase.ALIGN,
                PullCubeToolSkillPhase.PULL,
            ],
            [[0.02, 0.0, 0.025], [0.13, -0.067, 0.025], [-0.22, -0.067, 0.025]],
        ),
        (
            PushCubeSkillFSM,
            _PushCubeFSMEnv(num_envs=2),
            [PushCubeSkillPhase.ALIGN, PushCubeSkillPhase.PUSH],
            [[-0.025, 0.0, 0.02], [0.08, 0.1, 0.02]],
        ),
        (
            PullCubeSkillFSM,
            _PullCubeFSMEnv(num_envs=2),
            [PullCubeSkillPhase.ALIGN, PullCubeSkillPhase.PULL],
            [[0.03, 0.0, 0.02], [-0.15, -0.1, 0.02]],
        ),
        (
            RollBallSkillFSM,
            _RollBallFSMEnv(num_envs=2),
            [RollBallSkillPhase.ALIGN, RollBallSkillPhase.HIT],
            [[0.0, 0.085, 0.035], [0.0, 0.085, 0.035]],
        ),
    ]

    for fsm_cls, env, phases, expected_points in cases:
        fsm = fsm_cls(num_envs=len(phases), device="cpu")
        fsm.phase.copy_(torch.tensor([int(x) for x in phases], dtype=torch.long))
        normalized = normalize_skill_context(
            fsm.build_context(env), num_envs=len(phases), device="cpu"
        )

        assert normalized.target_pose_valid.tolist() == [True] * len(phases)
        assert normalized.target_point_valid.tolist() == [True] * len(phases)
        assert normalized.target_object == ["tcp"] * len(phases)
        assert torch.allclose(
            normalized.target_point_world,
            torch.tensor(expected_points, dtype=torch.float32),
            atol=1e-5,
        )
        assert torch.allclose(
            normalized.target_point_world,
            normalized.target_pose_world[:, :3, 3],
        )
        assert torch.allclose(
            normalized.target_pose_world[0, :3, 2],
            torch.tensor([0.0, 0.0, -1.0]),
            atol=1e-5,
        )
        assert torch.allclose(
            normalized.target_pose_world[0, :3, 1],
            torch.tensor([0.0, -1.0, 0.0]),
            atol=1e-5,
        )


def test_new_task_skill_fsm_tcp_target_geometry_and_task_meta():
    cases = [
        (
            LiftPegUprightSkillFSM,
            _LiftPegUprightFSMEnv(num_envs=4),
            [
                LiftPegUprightSkillPhase.PICK,
                LiftPegUprightSkillPhase.LIFT,
                LiftPegUprightSkillPhase.ROTATE,
                LiftPegUprightSkillPhase.LOWER,
            ],
            [
                [0.1, 0.0, 0.0],
                [-0.1, 0.0, 0.3],
                [-0.1, 0.0, 0.2],
                [-0.1, 0.0, 0.2],
            ],
            "LiftPegUpright-v1",
            {
                "success",
                "is_grasped",
                "is_peg_upright",
                "close_to_table",
                "safe_peg_height",
                "lift_height_error",
                "is_lifted",
                "upright_axis_error",
                "orientation_aligned",
                "z_error",
                "desired_peg_pose_world",
            },
        ),
        (
            PlugChargerSkillFSM,
            _PlugChargerFSMEnv(num_envs=3),
            [
                PlugChargerSkillPhase.PICK,
                PlugChargerSkillPhase.PRE_INSERT,
                PlugChargerSkillPhase.INSERT,
            ],
            [[-0.02, 0.0, 0.0], [0.05, 0.0, 0.45], [0.10, 0.0, 0.45]],
            "PlugCharger-v1",
            {
                "success",
                "is_grasped",
                "obj_to_goal_dist",
                "obj_to_goal_angle",
                "pre_inserted",
                "pre_x_error",
                "pre_yz_error",
                "pre_angle_error",
                "charger_base_pose_world",
                "object_goal_pose_world",
            },
        ),
        (
            PushTSkillFSM,
            _PushTFSMEnv(num_envs=2, tee_yaw=[0.2, 0.0]),
            [PushTSkillPhase.PUSH_ORIENT, PushTSkillPhase.PUSH_PLACE],
            [[0.1, -0.0375, 0.02], [0.095, 0.0, 0.02]],
            "PushT-v1",
            {
                "success",
                "intersection",
                "tee_to_goal_dist",
                "yaw_error",
                "orientation_aligned",
                "rotation_contact_side",
                "object_goal_pose_world",
            },
        ),
        (
            StackPyramidSkillFSM,
            _StackPyramidFSMEnv(num_envs=3),
            [
                StackPyramidSkillPhase.PUSH_BASE,
                StackPyramidSkillPhase.PICK_TOP,
                StackPyramidSkillPhase.PLACE_TOP,
            ],
            [[0.035, 0.0, 0.02], [0.0, 0.0, 0.02], [-0.05, 0.0, 0.24]],
            "StackPyramid-v1",
            {
                "success",
                "base_pair_ready",
                "is_cubeC_grasped",
                "is_C_on_A",
                "is_C_on_B",
                "is_C_static",
                "cubeC_goal_point_world",
            },
        ),
    ]

    for fsm_cls, env, phases, expected_points, task_name, meta_keys in cases:
        fsm = fsm_cls(num_envs=len(phases), device="cpu")
        fsm.phase.copy_(torch.tensor([int(x) for x in phases], dtype=torch.long))

        context = fsm.build_context(env)
        normalized = normalize_skill_context(
            context, num_envs=len(phases), device="cpu"
        )

        assert normalized.target_pose_valid.tolist() == [True] * len(phases)
        assert normalized.target_point_valid.tolist() == [True] * len(phases)
        assert normalized.target_object == ["tcp"] * len(phases)
        assert torch.allclose(
            normalized.target_point_world,
            torch.tensor(expected_points, dtype=torch.float32),
            atol=1e-5,
        )
        assert torch.allclose(
            normalized.target_point_world,
            normalized.target_pose_world[:, :3, 3],
        )
        assert context.task_meta["task"] == task_name
        assert context.task_meta["target_frame"] == "world"
        assert context.task_meta["target_entity"] == "tcp"
        assert meta_keys.issubset(context.task_meta.keys())

        selected_context = fsm.build_context(env, env_idx=torch.tensor([1]))
        assert selected_context.phase_id.tolist() == [1]
        for key in meta_keys:
            value = selected_context.task_meta[key]
            if torch.is_tensor(value):
                assert value.shape[0] == 1
            elif isinstance(value, list):
                assert len(value) == 1


def test_lift_peg_upright_place_phase_target_geometry():
    env = _LiftPegUprightFSMEnv(num_envs=3)
    peg_pos = torch.tensor(
        [
            [0.01, 0.02, 0.025],
            [0.03, -0.02, 0.22],
            [-0.04, 0.01, 0.22],
        ],
        dtype=torch.float32,
    )
    upright_q = torch.tensor([0.5, 0.5, -0.5, 0.5], dtype=torch.float32)
    peg_q = torch.tensor(
        [
            [0.70710677, 0.70710677, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            upright_q.tolist(),
        ],
        dtype=torch.float32,
    )
    env.peg.set_pose(peg_pos, peg_q)

    fsm = LiftPegUprightSkillFSM(num_envs=3, device="cpu")
    fsm.phase.copy_(
        torch.tensor(
            [
                int(LiftPegUprightSkillPhase.LIFT),
                int(LiftPegUprightSkillPhase.ROTATE),
                int(LiftPegUprightSkillPhase.LOWER),
            ],
            dtype=torch.long,
        )
    )
    context = fsm.build_context(env)

    expected_peg_pos = peg_pos.clone()
    expected_peg_pos[:, 2] = torch.tensor([0.22, 0.22, 0.12])
    expected_peg_q = peg_q.clone()
    expected_peg_q[1:] = upright_q
    expected_peg_pose = Pose.create_from_pq(
        p=expected_peg_pos, q=expected_peg_q
    ).to_transformation_matrix()
    current_peg_pose = env.peg.pose.to_transformation_matrix()
    current_tcp_pose = env.agent.tcp.pose.to_transformation_matrix()
    current_grasp_transform = torch.linalg.inv(current_peg_pose) @ current_tcp_pose
    expected_target_pose = expected_peg_pose @ current_grasp_transform
    expected_target_pose[1, :3, 3] = current_tcp_pose[1, :3, 3]

    assert torch.allclose(context.target_pose_world, expected_target_pose, atol=1e-5)
    assert torch.allclose(
        context.target_point_world, expected_target_pose[:, :3, 3], atol=1e-5
    )


def test_tabletop_skill_fsm_done_has_no_tcp_target():
    cases = [
        (PickCubeSkillFSM, _PickCubeFSMEnv(num_envs=2), PickCubeSkillPhase.DONE),
        (
            PlaceSphereSkillFSM,
            _PlaceSphereFSMEnv(num_envs=2),
            PlaceSphereSkillPhase.DONE,
        ),
        (
            StackCubeSkillFSM,
            _StackCubeFSMEnv(num_envs=2),
            StackCubeSkillPhase.DONE,
        ),
        (
            LiftPegUprightSkillFSM,
            _LiftPegUprightFSMEnv(num_envs=2),
            LiftPegUprightSkillPhase.DONE,
        ),
        (PokeCubeSkillFSM, _PokeCubeFSMEnv(num_envs=2), PokeCubeSkillPhase.DONE),
        (
            PegInsertionSideSkillFSM,
            _PegInsertionSideFSMEnv(num_envs=2),
            PegInsertionSideSkillPhase.DONE,
        ),
        (
            PlugChargerSkillFSM,
            _PlugChargerFSMEnv(num_envs=2),
            PlugChargerSkillPhase.DONE,
        ),
        (
            PullCubeToolSkillFSM,
            _PullCubeToolFSMEnv(num_envs=2),
            PullCubeToolSkillPhase.DONE,
        ),
        (PushCubeSkillFSM, _PushCubeFSMEnv(num_envs=2), PushCubeSkillPhase.DONE),
        (PullCubeSkillFSM, _PullCubeFSMEnv(num_envs=2), PullCubeSkillPhase.DONE),
        (RollBallSkillFSM, _RollBallFSMEnv(num_envs=2), RollBallSkillPhase.DONE),
        (PushTSkillFSM, _PushTFSMEnv(num_envs=2), PushTSkillPhase.DONE),
        (
            StackPyramidSkillFSM,
            _StackPyramidFSMEnv(num_envs=2),
            StackPyramidSkillPhase.DONE,
        ),
    ]

    for fsm_cls, env, done_phase in cases:
        fsm = fsm_cls(num_envs=2, device="cpu")
        fsm.phase.fill_(int(done_phase))
        context = fsm.build_context(env)

        assert torch.isnan(context.target_pose_world).all()
        assert torch.isnan(context.target_point_world).all()
        if context.target_gripper_width is not None:
            assert torch.isnan(context.target_gripper_width).all()
        assert context.target_object == [None, None]

        normalized = normalize_skill_context(context, num_envs=2, device="cpu")
        assert normalized.target_pose_valid.tolist() == [False, False]
        assert normalized.target_point_valid.tolist() == [False, False]
        assert normalized.target_gripper_width_valid.tolist() == [False, False]


def test_new_tabletop_skill_fsm_active_gripper_width_and_task_meta():
    cases = [
        (
            PullCubeToolSkillFSM,
            _PullCubeToolFSMEnv(num_envs=3),
            [
                PullCubeToolSkillPhase.PICK,
                PullCubeToolSkillPhase.ALIGN,
                PullCubeToolSkillPhase.PULL,
            ],
            "PullCubeTool-v1",
            {
                "success",
                "is_tool_grasped",
                "tool_positioned",
                "tool_positioning_dist",
                "cube_to_base_dist",
                "workspace_target_world",
            },
        ),
        (
            PushCubeSkillFSM,
            _PushCubeFSMEnv(num_envs=2),
            [PushCubeSkillPhase.ALIGN, PushCubeSkillPhase.PUSH],
            "PushCube-v1",
            {
                "success",
                "reached_push_pose",
                "tcp_to_push_dist",
                "obj_to_goal_dist",
                "is_obj_on_table",
                "push_direction_world",
                "object_goal_point_world",
            },
        ),
        (
            PullCubeSkillFSM,
            _PullCubeFSMEnv(num_envs=2),
            [PullCubeSkillPhase.ALIGN, PullCubeSkillPhase.PULL],
            "PullCube-v1",
            {
                "success",
                "reached_pull_pose",
                "tcp_to_pull_dist",
                "obj_to_goal_dist",
                "pull_direction_world",
                "object_goal_point_world",
            },
        ),
        (
            RollBallSkillFSM,
            _RollBallFSMEnv(num_envs=2),
            [RollBallSkillPhase.ALIGN, RollBallSkillPhase.HIT],
            "RollBall-v1",
            {
                "success",
                "reached_hit_pose",
                "ball_forward_speed",
                "is_ball_rolling_toward_goal",
                "tcp_to_hit_dist",
                "ball_to_goal_dist",
                "roll_direction_world",
                "object_goal_point_world",
            },
        ),
    ]

    for fsm_cls, env, phases, task_name, meta_keys in cases:
        fsm = fsm_cls(num_envs=len(phases), device="cpu")
        fsm.phase.copy_(torch.tensor([int(x) for x in phases], dtype=torch.long))

        context = fsm.build_context(env)
        normalized = normalize_skill_context(
            context, num_envs=len(phases), device="cpu"
        )

        assert normalized.target_gripper_width_valid.tolist() == [True] * len(phases)
        assert torch.allclose(
            normalized.target_gripper_width,
            torch.zeros(len(phases), dtype=torch.float32),
        )
        assert context.task_meta["task"] == task_name
        assert context.task_meta["target_frame"] == "world"
        assert context.task_meta["target_entity"] == "tcp"
        assert meta_keys.issubset(context.task_meta.keys())

        selected_context = fsm.build_context(env, env_idx=[1])
        assert selected_context.phase_id.tolist() == [1]
        assert selected_context.target_gripper_width.tolist() == [0.0]
        for key in meta_keys:
            value = selected_context.task_meta[key]
            if torch.is_tensor(value):
                assert value.shape[0] == 1


def test_requested_task_skill_fsm_gripper_width_semantics():
    plug_env = _PlugChargerFSMEnv(num_envs=3)
    plug_fsm = PlugChargerSkillFSM(num_envs=3, device="cpu")
    plug_fsm.phase.copy_(
        torch.tensor(
            [
                int(PlugChargerSkillPhase.PICK),
                int(PlugChargerSkillPhase.PRE_INSERT),
                int(PlugChargerSkillPhase.INSERT),
            ],
            dtype=torch.long,
        )
    )
    plug_context = normalize_skill_context(
        plug_fsm.build_context(plug_env), num_envs=3, device="cpu"
    )
    assert plug_context.target_gripper_width_valid.tolist() == [True, True, True]
    assert torch.allclose(plug_context.target_gripper_width, torch.zeros(3))

    lift_env = _LiftPegUprightFSMEnv(num_envs=4)
    lift_fsm = LiftPegUprightSkillFSM(num_envs=4, device="cpu")
    lift_fsm.phase.copy_(
        torch.tensor(
            [
                int(LiftPegUprightSkillPhase.PICK),
                int(LiftPegUprightSkillPhase.LIFT),
                int(LiftPegUprightSkillPhase.ROTATE),
                int(LiftPegUprightSkillPhase.LOWER),
            ],
            dtype=torch.long,
        )
    )
    lift_context = normalize_skill_context(
        lift_fsm.build_context(lift_env), num_envs=4, device="cpu"
    )
    assert lift_context.target_gripper_width_valid.tolist() == [
        False,
        True,
        True,
        True,
    ]
    assert lift_context.target_gripper_width.tolist() == [0.0, 0.0, 0.0, 0.0]

    stack_env = _StackPyramidFSMEnv(num_envs=3)
    stack_fsm = StackPyramidSkillFSM(num_envs=3, device="cpu")
    stack_fsm.phase.copy_(
        torch.tensor(
            [
                int(StackPyramidSkillPhase.PUSH_BASE),
                int(StackPyramidSkillPhase.PICK_TOP),
                int(StackPyramidSkillPhase.PLACE_TOP),
            ],
            dtype=torch.long,
        )
    )
    stack_context = normalize_skill_context(
        stack_fsm.build_context(stack_env), num_envs=3, device="cpu"
    )
    assert stack_context.target_gripper_width_valid.tolist() == [True, False, True]
    assert torch.allclose(
        stack_context.target_gripper_width,
        torch.tensor([0.0, 0.0, 0.08], dtype=torch.float32),
    )

    push_t_context = PushTSkillFSM(num_envs=2, device="cpu").build_context(
        _PushTFSMEnv(num_envs=2)
    )
    assert push_t_context.target_gripper_width is None


@pytest.mark.parametrize(
    "env_id",
    [
        "PickCube-v1",
        "PlaceSphere-v1",
        "StackCube-v1",
        "LiftPegUpright-v1",
        "PokeCube-v1",
        "PegInsertionSide-v1",
        "PlugCharger-v1",
        "PullCubeTool-v1",
        "PushCube-v1",
        "PullCube-v1",
        "RollBall-v1",
        "PushT-v1",
        "StackPyramid-v1",
    ],
)
def test_tabletop_skill_annotation_context_smoke_after_reset(env_id):
    import gymnasium as gym

    env = gym.make(
        env_id,
        num_envs=1,
        obs_mode="state",
        sim_backend="cpu",
        render_backend="none",
    )
    try:
        env.reset()
        base_env = env.unwrapped
        context = base_env.get_skill_annotation_context()
        normalized = normalize_skill_context(
            context, num_envs=base_env.num_envs, device=base_env.device
        )

        assert normalized.skill_id.shape == (1,)
        assert normalized.phase_id.tolist() == [0]
        assert normalized.target_point_world.shape == (1, 3)
        assert normalized.target_point_valid.tolist() == [True]
        assert normalized.target_pose_valid.tolist() == [True]
        assert normalized.target_object == ["tcp"]
        assert torch.allclose(
            normalized.target_point_world,
            normalized.target_pose_world[:, :3, 3],
        )
        assert normalized.task_meta["task"] == env_id
    finally:
        env.close()


def test_reset_skill_annotator_accepts_supported_env_idx_types():
    env = _PickCubeStyleProviderEnv()

    for env_idx in (1, [1], (1,), np.array([1]), torch.tensor([1])):
        get_annotation_bundle(env)
        manager = env._skill_annotation_manager
        assert manager._previous[0] is not None
        assert manager._previous[1] is not None

        reset_skill_annotator(env, env_idx)

        assert manager._previous[0] is not None
        assert manager._previous[1] is None


def test_reset_skill_annotator_resets_task_annotation_fsm():
    env = _PickCubeStyleProviderEnv()

    class _FakeSkillFSM:
        def __init__(self):
            self.reset_calls = []

        def reset(self, env_idx=None):
            self.reset_calls.append(env_idx)

    fsm = _FakeSkillFSM()
    env._skill_annotation_fsm = fsm
    env_idx = np.array([1])

    reset_skill_annotator(env, env_idx)

    assert len(fsm.reset_calls) == 1
    np.testing.assert_array_equal(fsm.reset_calls[0], env_idx)


def test_skill_annotation_episode_recorder_flushes_t_plus_one_and_projection(tmp_path):
    import h5py

    env = _ProjectingSkillAnnotationEnv()
    recorder = SkillAnnotationEpisodeRecorder(cameras=["base_camera"])

    recorder.reset(env)
    assert len(recorder) == 1
    recorder.step(env)
    assert len(recorder) == 2

    path = tmp_path / "annotations.h5"
    with h5py.File(path, "w") as h5_file:
        group = h5_file.create_group("traj_0")
        recorder.flush_to_h5(group, start_ptr=0, end_ptr=2, env_idx=1)

        annotations = group["skill_annotations"]
        assert "skill" not in annotations
        assert "phase" not in annotations
        assert "skill_vocab" in annotations.attrs
        assert annotations["skill_id"].shape == (2,)
        assert annotations["skill_id"][:].tolist() == [SKILL_IDS["place"], SKILL_IDS["place"]]
        assert annotations["phase_id"].shape == (2,)
        assert annotations["phase_id"][:].tolist() == [20, 21]
        assert annotations["target"]["point_world"].shape == (2, 3)
        assert annotations["target"]["pose_world"].shape == (2, 4, 4)
        assert annotations["projection"]["base_camera"]["point_uv"].shape == (2, 2)
        assert annotations["projection"]["base_camera"]["point_visible"].shape == (2,)
        assert annotations["projection"]["base_camera"]["grasp_rect_uv"].shape == (2, 4, 2)
        assert annotations["projection"]["base_camera"]["grasp_visible"].shape == (2,)


def test_skill_annotation_episode_recorder_partial_reset_replaces_selected_env():
    env = _ProjectingSkillAnnotationEnv()
    recorder = SkillAnnotationEpisodeRecorder()

    recorder.reset(env)
    recorder.step(env)
    recorder.reset(env, env_idx=np.array([0]))

    assert len(recorder) == 2
    assert recorder.buffer["phase_id"][:, 0].tolist() == [10, 12]
    assert recorder.buffer["phase_id"][:, 1].tolist() == [20, 21]


def test_skill_annotation_episode_recorder_partial_reset_only_queries_selected_env():
    env = _ProjectingSkillAnnotationEnv()
    recorder = SkillAnnotationEpisodeRecorder()

    recorder.reset(env)
    recorder.step(env)
    manager = env._skill_annotation_manager
    assert manager._previous[0].phase_id == 11
    assert manager._previous[1].phase_id == 21

    env.requested_env_indices.clear()
    recorder.reset(env, env_idx=np.array([0]))

    assert env.requested_env_indices == [(0,)]
    assert manager._previous[0].phase_id == 12
    assert manager._previous[1].phase_id == 21


def test_skill_annotation_episode_recorder_partial_reset_can_initialize_buffer():
    env = _ProjectingSkillAnnotationEnv()
    recorder = SkillAnnotationEpisodeRecorder()

    recorder.reset(env, env_idx=np.array([1]))

    assert env.requested_env_indices == [(1,)]
    assert recorder.buffer["phase_id"].shape == (1, 2)
    assert recorder.buffer["phase_id"][0, 0] == -1
    assert recorder.buffer["phase_id"][0, 1] == 20

    recorder.step(env)

    assert recorder.buffer["phase_id"].shape == (2, 2)
    assert recorder.buffer["phase_id"][:, 1].tolist() == [20, 21]


def test_record_episode_writes_skill_annotations_when_enabled(tmp_path):
    import h5py

    from mani_skill.utils.wrappers import RecordEpisode

    env = RecordEpisode(
        _make_record_skill_annotation_env(),
        output_dir=str(tmp_path),
        trajectory_name="enabled",
        save_video=False,
        clean_on_close=False,
        record_skill_annotations=True,
    )

    env.reset()
    env.step(env.action_space.sample())
    env.close()

    with h5py.File(tmp_path / "enabled.h5", "r") as h5_file:
        annotations = h5_file["traj_0"]["skill_annotations"]
        assert annotations["skill_id"].shape == (2,)
        assert annotations["phase_id"].shape == (2,)
        assert annotations["phase_id"][:].tolist() == [0, 1]


def test_record_episode_does_not_write_skill_annotations_by_default(tmp_path):
    import h5py

    from mani_skill.utils.wrappers import RecordEpisode

    env = RecordEpisode(
        _make_record_skill_annotation_env(),
        output_dir=str(tmp_path),
        trajectory_name="disabled",
        save_video=False,
        clean_on_close=False,
    )

    env.reset()
    env.step(env.action_space.sample())
    env.close()

    with h5py.File(tmp_path / "disabled.h5", "r") as h5_file:
        assert "skill_annotations" not in h5_file["traj_0"]


def _bool_tensor(value, num_envs):
    tensor = torch.as_tensor(value, dtype=torch.bool)
    if tensor.ndim == 0:
        tensor = tensor.repeat(num_envs)
    return tensor


def _yaw_quat(yaw, num_envs=None):
    yaw = torch.as_tensor(yaw, dtype=torch.float32).reshape(-1)
    if num_envs is not None and yaw.numel() == 1:
        yaw = yaw.repeat(num_envs)
    q = torch.zeros((yaw.shape[0], 4), dtype=torch.float32)
    q[:, 0] = torch.cos(yaw / 2)
    q[:, 3] = torch.sin(yaw / 2)
    return q


class _PoseStub:
    def __init__(self, p, q=None):
        self._pose = Pose.create_from_pq(
            p=torch.as_tensor(p, dtype=torch.float32),
            q=None if q is None else torch.as_tensor(q, dtype=torch.float32),
        )

    @property
    def p(self):
        return self._pose.p

    @property
    def q(self):
        return self._pose.q

    @property
    def raw_pose(self):
        return self._pose.raw_pose

    @property
    def device(self):
        return self._pose.device

    def __mul__(self, other):
        return self._pose * other

    def inv(self):
        return self._pose.inv()

    def to_transformation_matrix(self):
        return self._pose.to_transformation_matrix()


class _ActorStub:
    def __init__(self, p, q=None, static=True):
        self.pose = _PoseStub(p, q)
        self.static = _bool_tensor(static, self.pose.p.shape[0])

    def is_static(self, lin_thresh=1e-2, ang_thresh=1e-1):
        return self.static.clone()

    def set_pose(self, p, q=None):
        self.pose = _PoseStub(p, q)


class _RobotStub:
    def __init__(self, num_envs, base_p=None):
        if base_p is None:
            base_p = torch.zeros((num_envs, 3), dtype=torch.float32)
        self._links = [_ActorStub(base_p)]

    def get_links(self):
        return self._links


class _GraspingAgentStub:
    def __init__(self, num_envs, tcp_p=None, tcp_q=None, base_p=None):
        self.num_envs = num_envs
        self.is_grasped = torch.zeros(num_envs, dtype=torch.bool)
        self.max_angle_calls = []
        if tcp_p is None:
            tcp_p = torch.tensor([[-0.1, 0.0, 0.2]], dtype=torch.float32).repeat(
                num_envs, 1
            )
        if tcp_q is None:
            tcp_q = torch.tensor([[0.0, 1.0, 0.0, 0.0]], dtype=torch.float32).repeat(
                num_envs, 1
            )
        self.tcp = _ActorStub(tcp_p, tcp_q)
        self.robot = _RobotStub(num_envs, base_p=base_p)
        self._actor_grasped = {}

    def is_grasping(self, actor, max_angle=None):
        self.max_angle_calls.append(max_angle)
        return self._actor_grasped.get(id(actor), self.is_grasped)

    def set_tcp_pos(self, tcp_p):
        self.tcp = _ActorStub(tcp_p, self.tcp.pose.q)

    def set_actor_grasped(self, actor, is_grasped):
        self._actor_grasped[id(actor)] = _bool_tensor(is_grasped, self.num_envs)


class _PickCubeFSMEnv:
    device = "cpu"

    def __init__(self, num_envs=1):
        self.num_envs = num_envs
        self.cube = _ActorStub(torch.zeros((num_envs, 3)))
        self.goal_site = _ActorStub(
            torch.tensor([[0.2, 0.0, 0.1]], dtype=torch.float32).repeat(
                num_envs, 1
            )
        )
        self.agent = _GraspingAgentStub(num_envs)
        self.set_info()

    def set_info(self, is_grasped=False, is_obj_placed=False, success=False):
        self._info = {
            "is_grasped": _bool_tensor(is_grasped, self.num_envs),
            "is_obj_placed": _bool_tensor(is_obj_placed, self.num_envs),
            "success": _bool_tensor(success, self.num_envs),
        }

    def evaluate(self):
        return {key: value.clone() for key, value in self._info.items()}


class _PlaceSphereFSMEnv:
    device = "cpu"
    radius = 0.02
    block_half_size = [0.0025, 0.025, 0.025]

    def __init__(self, num_envs=1):
        self.num_envs = num_envs
        self.obj = _ActorStub(torch.zeros((num_envs, 3)))
        self.bin = _ActorStub(
            torch.tensor([[0.1, 0.0, 0.0025]], dtype=torch.float32).repeat(
                num_envs, 1
            )
        )
        self.agent = _GraspingAgentStub(num_envs)
        self.set_info()

    def set_info(
        self,
        is_obj_grasped=False,
        is_obj_on_bin=False,
        is_obj_static=False,
        success=False,
    ):
        self._info = {
            "is_obj_grasped": _bool_tensor(is_obj_grasped, self.num_envs),
            "is_obj_on_bin": _bool_tensor(is_obj_on_bin, self.num_envs),
            "is_obj_static": _bool_tensor(is_obj_static, self.num_envs),
            "success": _bool_tensor(success, self.num_envs),
        }

    def evaluate(self):
        return {key: value.clone() for key, value in self._info.items()}


class _StackCubeFSMEnv:
    device = "cpu"

    def __init__(self, num_envs=1):
        self.num_envs = num_envs
        self.cube_half_size = torch.tensor([0.02, 0.02, 0.02], dtype=torch.float32)
        self.cubeA = _ActorStub(torch.zeros((num_envs, 3)))
        self.cubeB = _ActorStub(
            torch.tensor([[0.1, 0.0, 0.02]], dtype=torch.float32).repeat(num_envs, 1)
        )
        self.agent = _GraspingAgentStub(num_envs)
        self.set_info()

    def set_info(
        self,
        is_cubeA_grasped=False,
        is_cubeA_on_cubeB=False,
        is_cubeA_static=False,
        success=False,
    ):
        self._info = {
            "is_cubeA_grasped": _bool_tensor(is_cubeA_grasped, self.num_envs),
            "is_cubeA_on_cubeB": _bool_tensor(is_cubeA_on_cubeB, self.num_envs),
            "is_cubeA_static": _bool_tensor(is_cubeA_static, self.num_envs),
            "success": _bool_tensor(success, self.num_envs),
        }

    def evaluate(self):
        return {key: value.clone() for key, value in self._info.items()}


class _LiftPegUprightFSMEnv:
    device = "cpu"
    peg_half_length = 0.12

    def __init__(self, num_envs=1):
        self.num_envs = num_envs
        peg_pos = torch.zeros((num_envs, 3), dtype=torch.float32)
        peg_q = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32).repeat(
            num_envs, 1
        )
        if num_envs > 1:
            peg_pos[1:, 2] = self.peg_half_length
            peg_q[1:] = torch.tensor([0.5, 0.5, -0.5, 0.5], dtype=torch.float32)
        self.peg = _ActorStub(peg_pos, peg_q)
        self.agent = _GraspingAgentStub(num_envs)
        self.set_info()

    def set_info(self, success=False):
        self._info = {"success": _bool_tensor(success, self.num_envs)}

    def set_grasped(self, is_grasped):
        self.agent.is_grasped = _bool_tensor(is_grasped, self.num_envs)

    def evaluate(self):
        return {key: value.clone() for key, value in self._info.items()}


class _StackPyramidFSMEnv:
    device = "cpu"

    def __init__(self, num_envs=1):
        self.num_envs = num_envs
        self.cube_half_size = torch.tensor([0.02, 0.02, 0.02], dtype=torch.float32)
        self.cubeA = _ActorStub(
            torch.tensor([[0.0, 0.0, 0.02]], dtype=torch.float32).repeat(num_envs, 1)
        )
        self.cubeB = _ActorStub(
            torch.tensor([[0.1, 0.0, 0.02]], dtype=torch.float32).repeat(num_envs, 1)
        )
        self.cubeC = _ActorStub(
            torch.tensor([[0.0, 0.0, 0.02]], dtype=torch.float32).repeat(num_envs, 1)
        )
        self.agent = _GraspingAgentStub(num_envs)
        self.agent.set_actor_grasped(self.cubeA, False)
        self.agent.set_actor_grasped(self.cubeC, False)
        self.set_info()

    def set_info(self, success=False):
        self._info = {"success": _bool_tensor(success, self.num_envs)}

    def move_base_pair_ready(self):
        self.cubeA.set_pose(
            torch.tensor([[0.06, 0.0, 0.02]], dtype=torch.float32).repeat(
                self.num_envs, 1
            )
        )
        self.agent.set_actor_grasped(self.cubeA, False)
        self.cubeA.static = torch.ones(self.num_envs, dtype=torch.bool)

    def set_cubeC_grasped(self, is_grasped):
        self.agent.set_actor_grasped(self.cubeC, is_grasped)

    def evaluate(self):
        return {key: value.clone() for key, value in self._info.items()}


class _PokeCubeFSMEnv:
    device = "cpu"
    cube_half_size = 0.02

    def __init__(self, num_envs=1):
        self.num_envs = num_envs
        self.peg = _ActorStub(torch.zeros((num_envs, 3)))
        self.peg_head_offsets = _PoseStub(
            torch.tensor([[0.12, 0.0, 0.0]], dtype=torch.float32).repeat(
                num_envs, 1
            )
        )
        self.cube = _ActorStub(
            torch.tensor([[0.1, 0.0, self.cube_half_size]], dtype=torch.float32).repeat(
                num_envs, 1
            )
        )
        self.goal_region = _ActorStub(
            torch.tensor([[0.2, 0.0, 0.001]], dtype=torch.float32).repeat(
                num_envs, 1
            )
        )
        self.agent = _GraspingAgentStub(num_envs)
        self.set_info()

    def set_info(
        self,
        success=False,
        is_cube_placed=False,
        is_peg_cube_fit=False,
        is_peg_grasped=False,
        angle_diff=0.0,
        head_to_cube_dist=0.1,
    ):
        self._info = {
            "success": _bool_tensor(success, self.num_envs),
            "is_cube_placed": _bool_tensor(is_cube_placed, self.num_envs),
            "is_peg_cube_fit": _bool_tensor(is_peg_cube_fit, self.num_envs),
            "is_peg_grasped": _bool_tensor(is_peg_grasped, self.num_envs),
            "angle_diff": torch.as_tensor(angle_diff, dtype=torch.float32).reshape(-1),
            "head_to_cube_dist": torch.as_tensor(
                head_to_cube_dist, dtype=torch.float32
            ).reshape(-1),
        }
        for key in ("angle_diff", "head_to_cube_dist"):
            if self._info[key].numel() == 1:
                self._info[key] = self._info[key].repeat(self.num_envs)

    def evaluate(self):
        return {key: value.clone() for key, value in self._info.items()}


class _PegInsertionSideFSMEnv:
    device = "cpu"

    def __init__(self, num_envs=1):
        self.num_envs = num_envs
        self.peg = _ActorStub(torch.zeros((num_envs, 3)))
        self.peg_half_sizes = torch.tensor(
            [[0.1, 0.02, 0.02]], dtype=torch.float32
        ).repeat(num_envs, 1)
        self.goal_pose = _PoseStub(
            torch.tensor([[0.2, 0.0, 0.05]], dtype=torch.float32).repeat(
                num_envs, 1
            )
        )
        self.agent = _GraspingAgentStub(num_envs)
        self.set_info()

    def set_info(self, success=False, is_grasped=False, pre_inserted=False):
        self.agent.is_grasped = _bool_tensor(is_grasped, self.num_envs)
        self.pre_inserted = _bool_tensor(pre_inserted, self.num_envs)
        self._info = {
            "success": _bool_tensor(success, self.num_envs),
            "peg_head_pos_at_hole": torch.zeros(
                (self.num_envs, 3), dtype=torch.float32
            ),
        }

    def evaluate(self):
        return {key: value.clone() for key, value in self._info.items()}

    def get_peg_pre_insertion_info(self):
        distance = torch.zeros(self.num_envs, dtype=torch.float32)
        return self.pre_inserted, distance, distance


class _PlugChargerFSMEnv:
    device = "cpu"
    _base_size = [0.02, 0.015, 0.012]

    def __init__(self, num_envs=1):
        self.num_envs = num_envs
        self.charger = _ActorStub(torch.zeros((num_envs, 3), dtype=torch.float32))
        self.goal_pose = _PoseStub(
            torch.tensor([[0.2, 0.0, 0.25]], dtype=torch.float32).repeat(
                num_envs, 1
            )
        )
        self.agent = _GraspingAgentStub(num_envs)
        self.set_info()

    @property
    def charger_base_pose(self):
        return self.charger.pose._pose * Pose.create_from_pq(
            p=torch.tensor([-self._base_size[0], 0.0, 0.0], dtype=torch.float32)
        )

    def set_info(self, success=False, obj_to_goal_dist=0.1, obj_to_goal_angle=0.5):
        self._info = {
            "success": _bool_tensor(success, self.num_envs),
            "obj_to_goal_dist": torch.as_tensor(
                obj_to_goal_dist, dtype=torch.float32
            ).reshape(-1),
            "obj_to_goal_angle": torch.as_tensor(
                obj_to_goal_angle, dtype=torch.float32
            ).reshape(-1),
        }
        for key in ("obj_to_goal_dist", "obj_to_goal_angle"):
            if self._info[key].numel() == 1:
                self._info[key] = self._info[key].repeat(self.num_envs)

    def set_grasped(self, is_grasped):
        self.agent.is_grasped = _bool_tensor(is_grasped, self.num_envs)

    def move_charger_to_pre_insert_pose(self):
        pre_insert_pose = self.goal_pose._pose * Pose.create_from_pq(
            p=torch.tensor([-0.05, 0.0, 0.0], dtype=torch.float32)
        )
        self.charger.set_pose(pre_insert_pose.p, pre_insert_pose.q)

    def evaluate(self):
        return {key: value.clone() for key, value in self._info.items()}


class _PullCubeToolFSMEnv:
    device = "cpu"
    cube_half_size = 0.02
    hook_length = 0.05

    def __init__(self, num_envs=1):
        self.num_envs = num_envs
        self.cube = _ActorStub(
            torch.tensor([[0.2, 0.0, 0.025]], dtype=torch.float32).repeat(
                num_envs, 1
            )
        )
        self.l_shape_tool = _ActorStub(
            torch.tensor([[0.0, 0.0, 0.025]], dtype=torch.float32).repeat(
                num_envs, 1
            )
        )
        self.agent = _GraspingAgentStub(num_envs)
        self.set_info()

    def set_info(self, success=False):
        self._info = {
            "success": _bool_tensor(success, self.num_envs),
            "success_once": _bool_tensor(success, self.num_envs),
            "success_at_end": _bool_tensor(success, self.num_envs),
        }

    def set_tool_grasped(self, is_grasped):
        self.agent.is_grasped = _bool_tensor(is_grasped, self.num_envs)

    def set_tool_positioned(self, positioned):
        positioned = _bool_tensor(positioned, self.num_envs)
        align_target = self.cube.pose.p + torch.tensor(
            [-(self.hook_length + self.cube_half_size), -0.067, 0],
            dtype=torch.float32,
        )
        default_tcp_pos = torch.tensor(
            [[-0.1, 0.0, 0.2]], dtype=torch.float32
        ).repeat(self.num_envs, 1)
        tcp_pos = torch.where(positioned[:, None], align_target, default_tcp_pos)
        self.agent.set_tcp_pos(tcp_pos)

    def evaluate(self):
        return {key: value.clone() for key, value in self._info.items()}


class _PushCubeFSMEnv:
    device = "cpu"
    cube_half_size = 0.02

    def __init__(self, num_envs=1):
        self.num_envs = num_envs
        self.obj = _ActorStub(
            torch.tensor([[0.0, 0.0, self.cube_half_size]], dtype=torch.float32).repeat(
                num_envs, 1
            )
        )
        self.goal_region = _ActorStub(
            torch.tensor([[0.2, 0.1, 0.001]], dtype=torch.float32).repeat(
                num_envs, 1
            )
        )
        self.agent = _GraspingAgentStub(num_envs)
        self.set_info()

    def set_info(self, success=False):
        self._info = {"success": _bool_tensor(success, self.num_envs)}

    def move_tcp_to_push_pose(self):
        self.agent.set_tcp_pos(
            self.obj.pose.p
            + torch.tensor([-self.cube_half_size - 0.005, 0, 0], dtype=torch.float32)
        )

    def evaluate(self):
        return {key: value.clone() for key, value in self._info.items()}


class _PullCubeFSMEnv:
    device = "cpu"
    cube_half_size = 0.02

    def __init__(self, num_envs=1):
        self.num_envs = num_envs
        self.obj = _ActorStub(
            torch.tensor([[0.0, 0.0, self.cube_half_size]], dtype=torch.float32).repeat(
                num_envs, 1
            )
        )
        self.goal_region = _ActorStub(
            torch.tensor([[-0.2, -0.1, 0.001]], dtype=torch.float32).repeat(
                num_envs, 1
            )
        )
        self.agent = _GraspingAgentStub(num_envs)
        self.set_info()

    def set_info(self, success=False):
        self._info = {"success": _bool_tensor(success, self.num_envs)}

    def move_tcp_to_pull_pose(self):
        self.agent.set_tcp_pos(
            self.obj.pose.p
            + torch.tensor([self.cube_half_size + 0.01, 0, 0], dtype=torch.float32)
        )

    def evaluate(self):
        return {key: value.clone() for key, value in self._info.items()}


class _RollBallFSMEnv:
    device = "cpu"
    ball_radius = 0.035

    def __init__(self, num_envs=1):
        self.num_envs = num_envs
        self.ball = _ActorStub(
            torch.tensor([[0.0, 0.0, self.ball_radius]], dtype=torch.float32).repeat(
                num_envs, 1
            )
        )
        self.ball.linear_velocity = torch.zeros(
            (num_envs, 3), dtype=torch.float32
        )
        self.goal_region = _ActorStub(
            torch.tensor([[0.0, -0.2, 0.001]], dtype=torch.float32).repeat(
                num_envs, 1
            )
        )
        self.agent = _GraspingAgentStub(num_envs)
        self.reached_status = torch.zeros(num_envs, dtype=torch.float32)
        self.set_info()

    def set_info(self, success=False):
        self._info = {"success": _bool_tensor(success, self.num_envs)}

    def set_ball_velocity(self, velocity):
        velocity = torch.as_tensor(velocity, dtype=torch.float32).reshape(-1, 3)
        if velocity.shape[0] == 1:
            velocity = velocity.repeat(self.num_envs, 1)
        self.ball.linear_velocity = velocity

    def move_tcp_to_hit_pose(self):
        direction_xy = self.goal_region.pose.p[:, :2] - self.ball.pose.p[:, :2]
        direction_xy = direction_xy / torch.linalg.norm(
            direction_xy, dim=1, keepdim=True
        )
        direction = torch.zeros_like(self.ball.pose.p)
        direction[:, :2] = direction_xy
        self.agent.set_tcp_pos(
            self.ball.pose.p - direction * (self.ball_radius + 0.05)
        )

    def evaluate(self):
        return {key: value.clone() for key, value in self._info.items()}


class _PushTFSMEnv:
    device = "cpu"
    intersection_thresh = 0.90

    def __init__(self, num_envs=1, tee_yaw=0.2):
        self.num_envs = num_envs
        tee_pos = torch.tensor([[0.0, 0.0, 0.02]], dtype=torch.float32).repeat(
            num_envs, 1
        )
        goal_pos = torch.tensor([[0.2, 0.0, 0.001]], dtype=torch.float32).repeat(
            num_envs, 1
        )
        self.tee = _ActorStub(tee_pos, _yaw_quat(tee_yaw, num_envs=num_envs))
        self.goal_tee = _ActorStub(goal_pos, _yaw_quat(0.0, num_envs=num_envs))
        self.agent = _GraspingAgentStub(num_envs)
        self.set_intersection(0.0)

    def quat_to_z_euler(self, quats):
        signs = torch.ones_like(quats[:, -1])
        signs[quats[:, -1] < 0] = -1.0
        qw = quats[:, 0] * signs
        return 2 * qw.acos()

    def set_tee_yaw(self, yaw):
        self.tee.set_pose(self.tee.pose.p, _yaw_quat(yaw, num_envs=self.num_envs))

    def set_intersection(self, intersection):
        self.intersection = torch.as_tensor(intersection, dtype=torch.float32).reshape(
            -1
        )
        if self.intersection.numel() == 1:
            self.intersection = self.intersection.repeat(self.num_envs)

    def pseudo_render_intersection(self):
        return self.intersection.clone()

    def evaluate(self):
        return {"success": self.pseudo_render_intersection() >= self.intersection_thresh}


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
            indices = torch.as_tensor(env_idx, dtype=torch.long).reshape(-1)

        is_grasped = self.is_grasped[indices]
        target_point = torch.where(
            is_grasped[:, None], self.goal_pos[indices], self.cube_pos[indices]
        )
        target_pose = torch.eye(4)[None].repeat(len(indices), 1, 1)
        target_pose[:, :3, 3] = target_point
        skills = ["place" if bool(item) else "pick" for item in is_grasped]
        phases = ["transport" if bool(item) else "approach" for item in is_grasped]
        phase_ids = torch.where(
            is_grasped,
            torch.full((len(indices),), 20, dtype=torch.long),
            torch.full((len(indices),), 10, dtype=torch.long),
        )

        return SkillAnnotationContext(
            skill=skills,
            skill_state=phases,
            phase_id=phase_ids,
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


class _ProjectingSkillAnnotationEnv:
    num_envs = 2
    device = "cpu"

    def __init__(self):
        self._call_idx = 0
        self.requested_env_indices = []

    def get_skill_annotation_context(self, env_idx=None):
        if env_idx is None:
            self.requested_env_indices.append(None)
            indices = torch.arange(self.num_envs)
        elif torch.is_tensor(env_idx):
            self.requested_env_indices.append(
                tuple(int(i) for i in env_idx.detach().cpu().flatten().tolist())
            )
            indices = env_idx.flatten().long()
        else:
            self.requested_env_indices.append(
                tuple(int(i) for i in torch.as_tensor(env_idx).flatten().tolist())
            )
            indices = torch.as_tensor(env_idx, dtype=torch.long).reshape(-1)

        call_idx = self._call_idx
        self._call_idx += 1
        base_phase_ids = torch.tensor([10, 20], dtype=torch.long)
        target_point = torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [0.1, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )[indices]
        target_pose = torch.eye(4)[None].repeat(len(indices), 1, 1)
        target_pose[:, :3, 3] = target_point
        skills = ["pick" if int(idx) == 0 else "place" for idx in indices]

        return SkillAnnotationContext(
            skill=skills,
            phase_id=base_phase_ids[indices] + call_idx,
            target_point_world=target_point,
            target_pose_world=target_pose,
            target_gripper_width=torch.full((len(indices),), 0.04),
        )

    def get_sensor_params(self):
        intrinsic = torch.tensor(
            [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
        extrinsic = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
            dtype=torch.float32,
        )
        return {
            "base_camera": {
                "intrinsic_cv": intrinsic[None].repeat(self.num_envs, 1, 1),
                "extrinsic_cv": extrinsic[None].repeat(self.num_envs, 1, 1),
                "image_size": (100, 100),
            }
        }


def _make_record_skill_annotation_env():
    import gymnasium as gym

    class RecordSkillAnnotationEnv(gym.Env):
        num_envs = 1
        device = "cpu"
        control_mode = "mock"

        def __init__(self):
            self.t = 0
            self._episode_seed = [0]
            self.single_action_space = gym.spaces.Box(
                low=-1.0, high=1.0, shape=(1,), dtype=np.float32
            )
            self.action_space = self.single_action_space
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
            )

        def reset(self, *, seed=None, options=None):
            self.t = 0
            return np.array([float(self.t)], dtype=np.float32), {"reconfigure": False}

        def step(self, action):
            self.t += 1
            obs = np.array([float(self.t)], dtype=np.float32)
            return obs, 0.0, False, False, {}

        def get_state_dict(self):
            return {
                "actors": {
                    "mock": np.array([[float(self.t)]], dtype=np.float32),
                }
            }

        def get_skill_annotation_context(self, env_idx=None):
            target_pose = torch.eye(4)[None]
            target_pose[:, 2, 3] = 1.0
            return SkillAnnotationContext(
                skill="pick",
                phase_id=self.t,
                target_point_world=torch.tensor([[float(self.t), 0.0, 1.0]]),
                target_pose_world=target_pose,
                target_gripper_width=torch.tensor([0.04]),
            )

    return RecordSkillAnnotationEnv()
