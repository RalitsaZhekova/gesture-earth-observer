from config import ZoomGestureConfig
from gesture_state import GestureMode, GestureSessionState, PinchModeController
from hand_geometry import DistanceMeasurement
from map_state import SharedMapState


def clear_zoom_tracking(gesture_state: GestureSessionState) -> None:
    gesture_state.zoom_anchor_distance = None
    gesture_state.zoom_anchor_level = None


def clamp_zoom_level(zoom_level: float, zoom_config: ZoomGestureConfig) -> float:
    return float(__import__("numpy").clip(zoom_level, zoom_config.min_zoom, zoom_config.max_zoom))


def update_zoom_mode(
    measurement: DistanceMeasurement,
    gesture_state: GestureSessionState,
    zoom_controller: PinchModeController,
    zoom_config: ZoomGestureConfig,
    shared_map_state: SharedMapState,
    now: float,
) -> None:
    was_zoom_active = gesture_state.active_mode == GestureMode.ZOOM
    if was_zoom_active:
        if measurement.length >= zoom_controller.toggle_config.release_distance:
            gesture_state.pinch_frames = 0
            gesture_state.pinch_armed = True
    else:
        zoom_controller.update_toggle(measurement.length, gesture_state)
    shared_map_state.update(active_mode=gesture_state.active_mode.value, updated_at=now)

    if gesture_state.active_mode != GestureMode.ZOOM:
        if was_zoom_active:
            clear_zoom_tracking(gesture_state)
        return

    snapshot = shared_map_state.snapshot()
    if not was_zoom_active or gesture_state.zoom_anchor_distance is None:
        gesture_state.zoom_anchor_distance = measurement.length
        gesture_state.zoom_anchor_level = float(snapshot["zoom_level"])

    snapshot = shared_map_state.snapshot()
    distance_delta = measurement.length - gesture_state.zoom_anchor_distance
    target_zoom = clamp_zoom_level(
        gesture_state.zoom_anchor_level + distance_delta * zoom_config.zoom_sensitivity,
        zoom_config,
    )
    current_zoom = float(snapshot["zoom_level"])
    smoothed_zoom = current_zoom + (target_zoom - current_zoom) * zoom_config.smoothing_factor

    if abs(smoothed_zoom - current_zoom) < zoom_config.zoom_deadband:
        smoothed_zoom = current_zoom

    shared_map_state.update(
        zoom_level=smoothed_zoom,
        target_zoom_level=target_zoom,
        active_mode=gesture_state.active_mode.value,
        updated_at=now,
    )
    if zoom_controller.should_exit_on_stable_hold(measurement.length, gesture_state, now):
        zoom_controller.exit_mode(gesture_state)
        clear_zoom_tracking(gesture_state)
        shared_map_state.update(active_mode=gesture_state.active_mode.value, updated_at=now)