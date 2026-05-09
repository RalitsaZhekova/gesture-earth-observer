import numpy as np

from config import PanGestureConfig
from gesture_state import GestureMode, GestureSessionState, GestureStateMachine, StableHoldExitConfig
from hand_geometry import DistanceMeasurement
from map_state import SharedMapState


def clear_pan_tracking(gesture_state: GestureSessionState) -> None:
    gesture_state.pan_pinch_frames = 0
    gesture_state.pan_pinch_armed = True
    gesture_state.pan_stable_value = None
    gesture_state.pan_stable_since = None
    gesture_state.pan_anchor_x = None
    gesture_state.pan_anchor_y = None
    gesture_state.pan_anchor_lat = None
    gesture_state.pan_anchor_lng = None


def pixels_to_geo_delta(
    delta_x: float,
    delta_y: float,
    center_lat: float,
    zoom_level: float,
    pan_config: PanGestureConfig,
) -> tuple[float, float]:
    pixels_per_world = 256 * (2**zoom_level)
    longitude_per_pixel = 360.0 / pixels_per_world
    latitude_per_pixel = longitude_per_pixel * np.cos(np.radians(center_lat))

    delta_lng = -delta_x * longitude_per_pixel * pan_config.pixels_to_longitude_scale
    delta_lat = delta_y * latitude_per_pixel * pan_config.pixels_to_longitude_scale
    return delta_lat, delta_lng


def smooth_value(current_value: float, target_value: float, smoothing_factor: float) -> float:
    return current_value + (target_value - current_value) * smoothing_factor


def update_pan_toggle(
    measurement_length: float,
    gesture_state: GestureSessionState,
    pan_config: PanGestureConfig,
    state_machine: GestureStateMachine,
    now: float,
) -> None:
    if gesture_state.active_mode == GestureMode.PAN:
        if measurement_length >= pan_config.release_distance:
            state_machine.exit_to_navigation(gesture_state, now)
        return

    if measurement_length <= pan_config.enter_distance:
        entered_pan = state_machine.enter_pan(gesture_state, now)
        if entered_pan:
            gesture_state.pan_stable_value = None
            gesture_state.pan_stable_since = None


def update_pan_mode(
    measurement: DistanceMeasurement,
    gesture_state: GestureSessionState,
    pan_config: PanGestureConfig,
    _exit_config: StableHoldExitConfig,
    state_machine: GestureStateMachine,
    shared_map_state: SharedMapState,
    now: float,
) -> None:
    was_pan_active = gesture_state.active_mode == GestureMode.PAN
    update_pan_toggle(measurement.length, gesture_state, pan_config, state_machine, now)
    shared_map_state.update(active_mode=gesture_state.active_mode.value, updated_at=now)

    if gesture_state.active_mode != GestureMode.PAN:
        if was_pan_active:
            clear_pan_tracking(gesture_state)
        return

    snapshot = shared_map_state.snapshot()
    pan_center = measurement.center
    if not was_pan_active or gesture_state.pan_anchor_x is None:
        gesture_state.pan_anchor_x = pan_center.x
        gesture_state.pan_anchor_y = pan_center.y
        gesture_state.pan_anchor_lat = float(snapshot["center_lat"])
        gesture_state.pan_anchor_lng = float(snapshot["center_lng"])

    delta_x = pan_center.x - gesture_state.pan_anchor_x
    delta_y = pan_center.y - gesture_state.pan_anchor_y
    movement = float(np.hypot(delta_x, delta_y))

    if movement >= pan_config.movement_deadband_pixels:
        target_lat_delta, target_lng_delta = pixels_to_geo_delta(
            delta_x,
            delta_y,
            gesture_state.pan_anchor_lat,
            float(snapshot["zoom_level"]),
            pan_config,
        )
        target_center_lat = gesture_state.pan_anchor_lat + target_lat_delta
        target_center_lng = gesture_state.pan_anchor_lng + target_lng_delta
        smoothed_center_lat = smooth_value(
            float(snapshot["center_lat"]),
            target_center_lat,
            pan_config.smoothing_factor,
        )
        smoothed_center_lng = smooth_value(
            float(snapshot["center_lng"]),
            target_center_lng,
            pan_config.smoothing_factor,
        )
        shared_map_state.update(
            center_lat=smoothed_center_lat,
            center_lng=smoothed_center_lng,
            active_mode=gesture_state.active_mode.value,
            updated_at=now,
        )
        gesture_state.pan_anchor_x = pan_center.x
        gesture_state.pan_anchor_y = pan_center.y
        gesture_state.pan_anchor_lat = smoothed_center_lat
        gesture_state.pan_anchor_lng = smoothed_center_lng