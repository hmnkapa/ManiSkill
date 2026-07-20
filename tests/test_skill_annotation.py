import numpy as np
import pytest
import torch

from mani_skill.envs.tasks.tabletop.peg_insertion_side import (
    PegInsertionSideSkillFSM,
    PegInsertionSideSkillPhase,
)
from mani_skill.envs.tasks.tabletop.place_sphere import (
    PlaceSphereSkillFSM,
    PlaceSphereSkillPhase,
)
from mani_skill.envs.tasks.tabletop.poke_cube import (
    PokeCubeSkillFSM,
    PokeCubeSkillPhase,
)
from mani_skill.envs.tasks.tabletop.stack_cube import (
    StackCubeSkillFSM,
    StackCubeSkillPhase,
)
from mani_skill.utils.skill_annotation import (
    SKILL_IDS,
    SKILL_NAMES,
    SKILL_VOCAB,
    SkillAnnotationEpisodeRecorder,
    SkillAnnotationContext,
    get_annotation_bundle,
    get_annotation_bundle_for_env,
    id_to_skill,
    normalize_skill_context,
    project_3d_to_2d,
    project_pose_to_grasp_annotation_2d,
    reset_skill_annotator,
    skill_to_id,
)


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
        grasp_height=0.2,
    )

    assert projected["grasp_rect_uv"].shape == (1, 4, 2)
    assert projected["grasp_visible"].tolist() == [True]
    expected = torch.tensor([[[40.0, 40.0], [60.0, 40.0], [60.0, 60.0], [40.0, 60.0]]])
    assert torch.allclose(projected["grasp_rect_uv"], expected)


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


def test_place_sphere_skill_fsm_transitions_wait_for_bin_settle():
    env = _PlaceSphereFSMEnv()
    fsm = PlaceSphereSkillFSM(num_envs=1, device="cpu")

    assert fsm.phase.tolist() == [int(PlaceSphereSkillPhase.PICK)]

    env.set_info(is_obj_grasped=True)
    fsm.update(env)
    assert fsm.phase.tolist() == [int(PlaceSphereSkillPhase.PLACE)]

    env.set_info(is_obj_grasped=False, is_obj_on_bin=True, success=False)
    fsm.update(env)
    assert fsm.phase.tolist() == [int(PlaceSphereSkillPhase.PLACE)]

    env.set_info(is_obj_grasped=False, is_obj_on_bin=False, success=False)
    fsm.update(env)
    assert fsm.phase.tolist() == [int(PlaceSphereSkillPhase.PICK)]

    env.set_info(is_obj_grasped=True)
    fsm.update(env)
    env.set_info(is_obj_grasped=False, is_obj_on_bin=True, success=True)
    fsm.update(env)
    assert fsm.phase.tolist() == [int(PlaceSphereSkillPhase.DONE)]

    env.set_info(is_obj_grasped=False, is_obj_on_bin=False, success=False)
    fsm.update(env)
    assert fsm.phase.tolist() == [int(PlaceSphereSkillPhase.DONE)]


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
    assert context.target_object == ["cube"]

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
    assert context.target_object == ["goal_region"]

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
    assert context.target_object == ["box_hole"]

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
    assert context.target_object == ["box_hole"]

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


def test_tabletop_skill_fsm_contexts_normalize_for_all_new_tasks():
    cases = [
        (
            PlaceSphereSkillFSM,
            _PlaceSphereFSMEnv(num_envs=2),
            [PlaceSphereSkillPhase.PICK, PlaceSphereSkillPhase.PLACE],
            [SKILL_IDS["pick"], SKILL_IDS["place"]],
            ["sphere", "bin"],
            [False, False],
            "PlaceSphere-v1",
        ),
        (
            StackCubeSkillFSM,
            _StackCubeFSMEnv(num_envs=2),
            [StackCubeSkillPhase.PICK, StackCubeSkillPhase.PLACE],
            [SKILL_IDS["pick"], SKILL_IDS["place"]],
            ["cubeA", "cubeB"],
            [False, False],
            "StackCube-v1",
        ),
        (
            PokeCubeSkillFSM,
            _PokeCubeFSMEnv(num_envs=2),
            [PokeCubeSkillPhase.PICK, PokeCubeSkillPhase.ALIGN],
            [SKILL_IDS["pick"], SKILL_IDS["push"]],
            ["peg", "cube"],
            [False, False],
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
            ["peg", "box_hole"],
            [False, True],
            "PegInsertionSide-v1",
        ),
    ]

    for fsm_cls, env, phases, skill_ids, target_objects, pose_valid, task_name in cases:
        fsm = fsm_cls(num_envs=2, device="cpu")
        fsm.phase.copy_(torch.tensor([int(x) for x in phases], dtype=torch.long))

        normalized = normalize_skill_context(
            fsm.build_context(env), num_envs=2, device="cpu"
        )

        assert normalized.skill_id.tolist() == skill_ids
        assert normalized.phase_id.tolist() == [0, 1]
        assert normalized.target_point_world.shape == (2, 3)
        assert normalized.target_point_valid.tolist() == [True, True]
        assert normalized.target_pose_valid.tolist() == pose_valid
        assert normalized.target_object == target_objects
        assert normalized.task_meta["task"] == task_name
        assert normalized.task_meta["target_frame"] == "world"


@pytest.mark.parametrize(
    "env_id",
    ["PlaceSphere-v1", "StackCube-v1", "PokeCube-v1", "PegInsertionSide-v1"],
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


class _PoseStub:
    def __init__(self, p):
        self.p = torch.as_tensor(p, dtype=torch.float32)
        if self.p.ndim == 1:
            self.p = self.p[None, :]

    def to_transformation_matrix(self):
        pose = torch.eye(4, dtype=torch.float32)[None].repeat(len(self.p), 1, 1)
        pose[:, :3, 3] = self.p
        return pose


class _ActorStub:
    def __init__(self, p):
        self.pose = _PoseStub(p)


class _GraspingAgentStub:
    def __init__(self, num_envs):
        self.num_envs = num_envs
        self.is_grasped = torch.zeros(num_envs, dtype=torch.bool)
        self.max_angle_calls = []

    def is_grasping(self, actor, max_angle=None):
        self.max_angle_calls.append(max_angle)
        return self.is_grasped


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


class _PokeCubeFSMEnv:
    device = "cpu"
    cube_half_size = 0.02

    def __init__(self, num_envs=1):
        self.num_envs = num_envs
        self.peg = _ActorStub(torch.zeros((num_envs, 3)))
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
