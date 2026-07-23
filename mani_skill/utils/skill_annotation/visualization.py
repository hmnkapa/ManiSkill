from __future__ import annotations

from typing import Any

import numpy as np
import torch

from mani_skill.utils.skill_annotation.schema import SKILL_NAME_TO_ID


GUIDANCE_POINT_COLOR_MAP = {
    "pick": (0, 255, 255),
    "screw": (0, 255, 255),
    "place": (255, 0, 0),
    "push": (255, 0, 0),
    "insert": (255, 0, 0),
}
GRASP_COLOR_GROUP_A = {"pick", "screw"}
GRASP_COLOR_GROUP_B = {"place", "push", "insert"}


def draw_skill_on_image(image: np.ndarray, skill: str | list[str] | None) -> np.ndarray:
    skill = _first_skill(skill)
    if skill not in SKILL_NAME_TO_ID or skill == "none":
        return image

    import cv2

    annotated = image.copy()
    text = f"skill: {skill}"
    origin = (10, 28)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.8
    thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    top_left = (6, 6)
    bottom_right = (top_left[0] + text_w + 12, top_left[1] + text_h + baseline + 12)
    cv2.rectangle(annotated, top_left, bottom_right, (0, 0, 0), thickness=-1)
    cv2.putText(
        annotated,
        text,
        origin,
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return annotated


def draw_guidance_point_on_image(
    image: np.ndarray,
    guidance_point_2d: Any,
    skill: str | list[str] | None = None,
    use_skill_color: bool = False,
) -> np.ndarray:
    uv = _extract_point_uv(guidance_point_2d)
    if uv is None:
        return image

    import cv2

    annotated = image.copy()
    skill_name = _first_skill(skill)
    point_color = (255, 0, 0)
    if use_skill_color and skill_name in GUIDANCE_POINT_COLOR_MAP:
        point_color = GUIDANCE_POINT_COLOR_MAP[skill_name]

    def _draw_point(frame: np.ndarray, center: tuple[int, int]) -> np.ndarray:
        overlay = frame.copy()
        cv2.circle(overlay, center, 2, point_color, thickness=-1, lineType=cv2.LINE_AA)
        return cv2.addWeighted(overlay, 0.5, frame, 0.5, 0.0)

    return _draw_on_single_or_single_batch(annotated, uv, _draw_point)


def draw_grasp_annotation_on_image(
    image: np.ndarray,
    grasp_annotation_2d: Any,
    skill: str | list[str] | None = None,
    use_skill_color: bool = False,
) -> np.ndarray:
    corners = _extract_grasp_rect_uv(grasp_annotation_2d)
    if corners is None:
        return image

    import cv2

    pts = np.round(corners).astype(np.int32)
    center = np.round(np.mean(corners, axis=0)).astype(np.int32)
    skill_name = _first_skill(skill)
    main_a_color = (255, 0, 0)
    main_b_color = (0, 0, 255)
    side_a_color = (0, 255, 255)
    side_b_color = (0, 255, 0)
    if use_skill_color and skill_name in GRASP_COLOR_GROUP_A:
        main_b_color = (0, 255, 0)
        side_a_color = (0, 0, 255)
        side_b_color = (255, 255, 0)
    elif use_skill_color and skill_name in GRASP_COLOR_GROUP_B:
        side_a_color = (0, 255, 0)
        side_b_color = (0, 255, 255)

    def _draw_rect(frame: np.ndarray) -> np.ndarray:
        overlay = frame.copy()
        cv2.line(overlay, tuple(pts[0]), tuple(pts[1]), main_a_color, 2, cv2.LINE_AA)
        cv2.line(overlay, tuple(pts[2]), tuple(pts[3]), main_b_color, 2, cv2.LINE_AA)
        cv2.line(overlay, tuple(pts[1]), tuple(pts[2]), side_b_color, 1, cv2.LINE_AA)
        cv2.line(overlay, tuple(pts[3]), tuple(pts[0]), side_a_color, 1, cv2.LINE_AA)
        cv2.circle(overlay, tuple(center), 2, (255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)
        return overlay

    if image.ndim == 4:
        if image.shape[0] != 1:
            return image
        annotated = image.copy()
        annotated[0] = _draw_rect(annotated[0])
        return annotated
    return _draw_rect(image.copy())


def _draw_on_single_or_single_batch(
    image: np.ndarray,
    uv: np.ndarray,
    draw_fn,
) -> np.ndarray:
    center = (int(uv[0]), int(uv[1]))
    if image.ndim == 4:
        if image.shape[0] != 1:
            return image
        frame = image[0]
        height, width = frame.shape[:2]
        if center[0] < 0 or center[0] >= width or center[1] < 0 or center[1] >= height:
            return image
        image[0] = draw_fn(frame, center)
        return image

    height, width = image.shape[:2]
    if center[0] < 0 or center[0] >= width or center[1] < 0 or center[1] >= height:
        return image
    return draw_fn(image, center)


def _extract_point_uv(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    visible = True
    if isinstance(value, dict):
        if "point_visible" in value:
            visible_arr = _to_numpy(value["point_visible"]).reshape(-1)
            visible = bool(visible_arr[0]) if visible_arr.size == 1 else False
        value = value.get("point_uv")
    uv = _to_numpy(value)
    if uv is None:
        return None
    if uv.ndim == 2:
        if uv.shape[0] != 1:
            return None
        uv = uv[0]
    if uv.shape != (2,) or not visible or np.any(uv < 0):
        return None
    return uv.astype(np.float32)


def _extract_grasp_rect_uv(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, dict):
        if not value:
            return None
        if value.get("style") == "grasp_rect":
            corners = value.get("corners")
            visible = True
        else:
            corners = value.get("grasp_rect_uv")
            visible = True
            if "grasp_visible" in value:
                visible_arr = _to_numpy(value["grasp_visible"]).reshape(-1)
                visible = bool(visible_arr[0]) if visible_arr.size == 1 else False
    else:
        corners = value
        visible = True
    corners = _to_numpy(corners)
    if corners is None:
        return None
    if corners.ndim == 3:
        if corners.shape[0] != 1:
            return None
        corners = corners[0]
    if corners.shape != (4, 2) or not visible or np.any(corners < 0):
        return None
    return corners.astype(np.float32)


def _to_numpy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _first_skill(skill: str | list[str] | None) -> str | None:
    if isinstance(skill, list):
        return skill[0] if skill else None
    return skill
