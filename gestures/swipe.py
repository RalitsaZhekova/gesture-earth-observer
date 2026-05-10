import numpy as np

from config import SwipeGestureConfig
from gesture_state import GestureMode, GestureSessionState, GestureStateMachine
from landmarks import (
    INDEX_MCP_ID,
    INDEX_PIP_ID,
    INDEX_TIP_ID,
    MIDDLE_MCP_ID,
    MIDDLE_PIP_ID,
    MIDDLE_TIP_ID,
    PINKY_MCP_ID,
    PINKY_PIP_ID,
    PINKY_TIP_ID,
    RING_PIP_ID,
    RING_TIP_ID,
    THUMB_IP_ID,
    THUMB_TIP_ID,
    WRIST_ID,
)
from map_state import SharedMapState


def to_landmark_map(landmarks: list[list[int]]) -> dict[int, tuple[int, int]]:
    return {landmark_id: (x, y) for landmark_id, x, y in landmarks}


def point_distance(point_a: tuple[int, int], point_b: tuple[int, int]) -> float:
    return float(np.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1]))


def clear_swipe_tracking(gesture_state: GestureSessionState, *, rearm: bool) -> None:
    gesture_state.swipe_open_frames = 0
    gesture_state.swipe_entry_x = None
    gesture_state.swipe_entry_y = None
    gesture_state.swipe_started_at = None
    gesture_state.swipe_last_open_at = None
    if rearm:
        gesture_state.swipe_armed = True


def is_open_palm(landmarks: list[list[int]]) -> bool:
    landmark_map = to_landmark_map(landmarks)
    required_ids = {
        WRIST_ID,
        THUMB_TIP_ID,
        THUMB_IP_ID,
        INDEX_MCP_ID,
        INDEX_PIP_ID,
        INDEX_TIP_ID,
        MIDDLE_MCP_ID,
        MIDDLE_PIP_ID,
        MIDDLE_TIP_ID,
        RING_PIP_ID,
        RING_TIP_ID,
        PINKY_MCP_ID,
        PINKY_PIP_ID,
        PINKY_TIP_ID,
    }
    if not required_ids.issubset(landmark_map):
        return False

    def finger_extended(tip_id: int, pip_id: int) -> bool:
        tip_y = landmark_map[tip_id][1]
        pip_y = landmark_map[pip_id][1]
        return tip_y < pip_y - 10

    fingers_extended = all(
        (
            finger_extended(INDEX_TIP_ID, INDEX_PIP_ID),
            finger_extended(MIDDLE_TIP_ID, MIDDLE_PIP_ID),
            finger_extended(RING_TIP_ID, RING_PIP_ID),
            finger_extended(PINKY_TIP_ID, PINKY_PIP_ID),
        )
    )
    if not fingers_extended:
        return False

    palm_width = point_distance(landmark_map[INDEX_MCP_ID], landmark_map[PINKY_MCP_ID])
    thumb_span = point_distance(landmark_map[THUMB_TIP_ID], landmark_map[INDEX_MCP_ID])
    wrist_to_middle = point_distance(landmark_map[WRIST_ID], landmark_map[MIDDLE_MCP_ID])
    thumb_open = thumb_span > palm_width * 0.55 and thumb_span > wrist_to_middle * 0.42
    return thumb_open


def update_swipe_mode(
    landmarks: list[list[int]],
    gesture_state: GestureSessionState,
    swipe_config: SwipeGestureConfig,
    state_machine: GestureStateMachine,
    shared_map_state: SharedMapState,
    now: float,
) -> bool:
    if gesture_state.active_mode in {GestureMode.ZOOM, GestureMode.PAN}:
        clear_swipe_tracking(gesture_state, rearm=True)
        return False

    palm_is_open = is_open_palm(landmarks)
    if not palm_is_open:
        if gesture_state.active_mode == GestureMode.SWIPE:
            if (
                gesture_state.swipe_last_open_at is not None
                and now - gesture_state.swipe_last_open_at <= swipe_config.lost_palm_grace_seconds
            ):
                shared_map_state.update(active_mode=gesture_state.active_mode.value, updated_at=now)
                return True
            state_machine.exit_to_navigation(gesture_state, now)
            clear_swipe_tracking(gesture_state, rearm=True)
            shared_map_state.update(active_mode=gesture_state.active_mode.value, updated_at=now)
            return False

        clear_swipe_tracking(gesture_state, rearm=True)
        shared_map_state.update(active_mode=gesture_state.active_mode.value, updated_at=now)
        return False

    gesture_state.swipe_open_frames += 1
    gesture_state.swipe_last_open_at = now
    if not gesture_state.swipe_armed:
        shared_map_state.update(active_mode=gesture_state.active_mode.value, updated_at=now)
        return False

    palm_center_x = int(np.mean([landmark[1] for landmark in landmarks]))
    palm_center_y = int(np.mean([landmark[2] for landmark in landmarks]))

    if gesture_state.active_mode != GestureMode.SWIPE:
        if gesture_state.swipe_open_frames < swipe_config.entry_frames:
            return False
        if not state_machine.enter_swipe(gesture_state, now):
            return False
        gesture_state.swipe_entry_x = palm_center_x
        gesture_state.swipe_entry_y = palm_center_y
        gesture_state.swipe_started_at = now
        shared_map_state.update(active_mode=gesture_state.active_mode.value, updated_at=now)
        return True

    shared_map_state.update(active_mode=gesture_state.active_mode.value, updated_at=now)
    if gesture_state.swipe_entry_x is None or gesture_state.swipe_entry_y is None:
        gesture_state.swipe_entry_x = palm_center_x
        gesture_state.swipe_entry_y = palm_center_y
        gesture_state.swipe_started_at = now
        return True

    if (
        gesture_state.swipe_started_at is not None
        and now - gesture_state.swipe_started_at > swipe_config.timeout_seconds
    ):
        gesture_state.swipe_entry_x = palm_center_x
        gesture_state.swipe_entry_y = palm_center_y
        gesture_state.swipe_started_at = now
        return True

    delta_x = palm_center_x - gesture_state.swipe_entry_x
    delta_y = palm_center_y - gesture_state.swipe_entry_y
    if abs(delta_y) > swipe_config.max_vertical_drift_pixels:
        gesture_state.swipe_entry_x = palm_center_x
        gesture_state.swipe_entry_y = palm_center_y
        gesture_state.swipe_started_at = now
        return True

    if abs(delta_x) < swipe_config.min_travel_pixels:
        return True

    swipe_triggered = (
        state_machine.request_swipe_right(gesture_state, now)
        if delta_x > 0
        else state_machine.request_swipe_left(gesture_state, now)
    )
    if swipe_triggered:
        clear_swipe_tracking(gesture_state, rearm=False)
        gesture_state.swipe_armed = False
        shared_map_state.update(active_mode=gesture_state.active_mode.value, updated_at=now)
    return swipe_triggered