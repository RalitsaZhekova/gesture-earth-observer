from config import ClassificationGestureConfig
from gesture_state import GestureSessionState
from gestures.swipe import is_open_palm
from map_state import SharedMapState


def clear_classification_tracking(gesture_state: GestureSessionState, *, rearm: bool) -> None:
    gesture_state.classification_frames = 0
    if rearm:
        gesture_state.classification_armed = True


def has_two_open_palms(hands_landmarks: list[list[list[int]]]) -> bool:
    open_palms = sum(1 for landmarks in hands_landmarks if is_open_palm(landmarks))
    return open_palms >= 2


def update_classification_gesture(
    hands_landmarks: list[list[list[int]]],
    gesture_state: GestureSessionState,
    classification_config: ClassificationGestureConfig,
    shared_map_state: SharedMapState,
    now: float,
) -> bool:
    if not has_two_open_palms(hands_landmarks):
        clear_classification_tracking(gesture_state, rearm=True)
        return False

    gesture_state.classification_frames += 1

    if (
        not gesture_state.classification_armed
        or gesture_state.classification_frames < classification_config.hold_frames
        or now < gesture_state.classification_locked_until
    ):
        return True

    snapshot = shared_map_state.snapshot()
    shared_map_state.update(
        classification_gesture_nonce=int(snapshot["classification_gesture_nonce"]) + 1,
        active_mode=gesture_state.active_mode.value,
        updated_at=now,
    )
    gesture_state.classification_armed = False
    gesture_state.classification_locked_until = now + classification_config.cooldown_seconds
    return True