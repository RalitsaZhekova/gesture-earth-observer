import time

import cv2
import numpy as np

from config import (
    CameraConfig,
    MapServerConfig,
    PanGestureConfig,
    SwipeGestureConfig,
    ZoomGestureConfig,
)
from gesture_state import (
    GestureMode,
    GestureSessionState,
    GestureStateMachine,
    PinchModeController,
    PinchToggleConfig,
    StableHoldExitConfig,
)
from hand_detector import HandDetector
from hand_geometry import (
    DistanceMeasurement,
    measure_distance_between_landmarks,
)
from gestures.feedback import draw_gesture_feedback, draw_ui
from gestures.pan import clear_pan_tracking, update_pan_mode
from gestures.swipe import clear_swipe_tracking, update_swipe_mode
from gestures.zoom import clear_zoom_tracking, update_zoom_mode
from landmarks import (
    INDEX_TIP_ID,
    MIDDLE_TIP_ID,
    THUMB_TIP_ID,
    WINDOW_NAME,
)
from map_state import SharedMapState, apply_pending_map_action
from map_server import open_map_browser, start_map_server


def create_camera(camera_config: CameraConfig) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(0)
    cap.set(3, camera_config.width)
    cap.set(4, camera_config.height)
    return cap

def publish_mode(
    gesture_state: GestureSessionState,
    shared_map_state: SharedMapState,
    now: float,
) -> None:
    shared_map_state.update(
        active_mode=gesture_state.active_mode.value,
        updated_at=now,
    )

def reset_tracking_when_no_hand(
    gesture_state: GestureSessionState,
    state_machine: GestureStateMachine,
    zoom_controller: PinchModeController,
    swipe_config: SwipeGestureConfig,
    shared_map_state: SharedMapState,
    now: float,
) -> None:
    swipe_grace_expired = (
        gesture_state.swipe_last_open_at is None
        or now - gesture_state.swipe_last_open_at > swipe_config.lost_palm_grace_seconds
    )

    if gesture_state.active_mode == GestureMode.SWIPE and swipe_grace_expired:
        state_machine.exit_to_navigation(gesture_state, now)

    zoom_controller.reset_tracking(gesture_state)
    clear_zoom_tracking(gesture_state)
    clear_pan_tracking(gesture_state)

    if gesture_state.active_mode != GestureMode.SWIPE:
        clear_swipe_tracking(gesture_state, rearm=True)

    publish_mode(gesture_state, shared_map_state, now)

def update_active_gesture(
    zoom_measurement: DistanceMeasurement,
    pan_measurement: DistanceMeasurement,
    gesture_state: GestureSessionState,
    zoom_controller: PinchModeController,
    zoom_config: ZoomGestureConfig,
    pan_config: PanGestureConfig,
    pan_exit_config: StableHoldExitConfig,
    state_machine: GestureStateMachine,
    shared_map_state: SharedMapState,
    now: float,
) -> DistanceMeasurement | None:
    if gesture_state.active_mode == GestureMode.SWIPE:
        clear_zoom_tracking(gesture_state)
        clear_pan_tracking(gesture_state)
        return None

    if gesture_state.active_mode == GestureMode.ZOOM:
        update_zoom_mode(
            zoom_measurement,
            gesture_state,
            zoom_controller,
            zoom_config,
            shared_map_state,
            now,
        )
        clear_pan_tracking(gesture_state)
        return zoom_measurement

    if gesture_state.active_mode == GestureMode.PAN:
        update_pan_mode(
            pan_measurement,
            gesture_state,
            pan_config,
            pan_exit_config,
            state_machine,
            shared_map_state,
            now,
        )
        clear_zoom_tracking(gesture_state)
        return pan_measurement

    update_zoom_mode(
        zoom_measurement,
        gesture_state,
        zoom_controller,
        zoom_config,
        shared_map_state,
        now,
    )

    if gesture_state.active_mode == GestureMode.ZOOM:
        clear_pan_tracking(gesture_state)
        return zoom_measurement

    update_pan_mode(
        pan_measurement,
        gesture_state,
        pan_config,
        pan_exit_config,
        state_machine,
        shared_map_state,
        now,
    )

    if gesture_state.active_mode == GestureMode.PAN:
        clear_zoom_tracking(gesture_state)
        return pan_measurement

    return None

def handle_landmarks(
    camera_frame: np.ndarray,
    landmarks: list[list[int]],
    gesture_state: GestureSessionState,
    state_machine: GestureStateMachine,
    zoom_controller: PinchModeController,
    zoom_config: ZoomGestureConfig,
    pan_config: PanGestureConfig,
    pan_exit_config: StableHoldExitConfig,
    swipe_config: SwipeGestureConfig,
    shared_map_state: SharedMapState,
    now: float,
) -> None:
    update_swipe_mode(
        landmarks,
        gesture_state,
        swipe_config,
        state_machine,
        shared_map_state,
        now,
    )

    zoom_measurement = measure_distance_between_landmarks(
        landmarks,
        THUMB_TIP_ID,
        INDEX_TIP_ID,
    )

    pan_measurement = measure_distance_between_landmarks(
        landmarks,
        INDEX_TIP_ID,
        MIDDLE_TIP_ID,
    )

    active_measurement = update_active_gesture(
        zoom_measurement,
        pan_measurement,
        gesture_state,
        zoom_controller,
        zoom_config,
        pan_config,
        pan_exit_config,
        state_machine,
        shared_map_state,
        now,
    )

    apply_pending_map_action(
        gesture_state,
        state_machine,
        shared_map_state,
        now,
    )

    draw_gesture_feedback(camera_frame, active_measurement, gesture_state)


def main() -> None:
    camera_config = CameraConfig()
    zoom_config = ZoomGestureConfig()
    pan_config = PanGestureConfig()
    swipe_config = SwipeGestureConfig()
    server_config = MapServerConfig()

    zoom_controller = PinchModeController(
        GestureMode.ZOOM,
        PinchToggleConfig(),
        StableHoldExitConfig(hold_seconds=1.5),
    )

    pan_exit_config = StableHoldExitConfig(hold_seconds=0.75)

    gesture_state = GestureSessionState()
    state_machine = GestureStateMachine()
    shared_map_state = SharedMapState()

    previous_time = 0.0

    server = start_map_server(server_config, shared_map_state)
    open_map_browser(server_config)

    cap = create_camera(camera_config)
    detector = HandDetector()

    try:
        while True:
            success, camera_frame = cap.read()

            if not success:
                continue

            now = time.time()

            state_machine.initialize_viewer(gesture_state, now)
            publish_mode(gesture_state, shared_map_state, now)

            camera_frame = cv2.flip(camera_frame, 1)
            camera_frame = detector.find_hands(camera_frame)

            landmarks = detector.find_position(
                camera_frame,
                landmark_ids=[
                    THUMB_TIP_ID,
                    INDEX_TIP_ID,
                    MIDDLE_TIP_ID,
                ],
            )

            if landmarks:
                handle_landmarks(
                    camera_frame,
                    landmarks,
                    gesture_state,
                    state_machine,
                    zoom_controller,
                    zoom_config,
                    pan_config,
                    pan_exit_config,
                    swipe_config,
                    shared_map_state,
                    now,
                )
            else:
                reset_tracking_when_no_hand(
                    gesture_state,
                    state_machine,
                    zoom_controller,
                    swipe_config,
                    shared_map_state,
                    now,
                )

            map_snapshot = shared_map_state.snapshot()

            previous_time = draw_ui(
                camera_frame,
                gesture_state,
                map_snapshot,
                previous_time,
                now,
            )

            cv2.imshow(WINDOW_NAME, camera_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        server.shutdown()
        server.server_close()

if __name__ == "__main__":
    main()