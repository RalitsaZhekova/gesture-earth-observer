from dataclasses import dataclass
import time

import cv2
import numpy as np
from pycaw.pycaw import AudioUtilities

from gesture_state import (
    GestureMode,
    GestureSessionState,
    PinchModeController,
    PinchToggleConfig,
    StableHoldExitConfig
)

from hand_detector import HandDetector
from hand_geometry import DistanceMeasurement, measure_distance_between_landmarks


THUMB_TIP_ID = 4
INDEX_TIP_ID = 8
WINDOW_NAME = "Image"


@dataclass(frozen=True)
class CameraConfig:
    width = 640
    height = 480


@dataclass(frozen=True)
class VolumeMappingConfig:
    min_distance: int = 30
    max_distance: int = 280
    min_bar_y: int = 150
    max_bar_y: int = 400


@dataclass(frozen=True)
class VolumeRange:
    minimum: int
    maximum: int


@dataclass
class VolumeUiState:
    volume_bar_y = 400


def create_camera(camera_config: CameraConfig) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(0)
    cap.set(3, camera_config.width)
    cap.set(4, camera_config.height)
    return cap

def get_system_volume():
    return AudioUtilities.GetSpeakers().EndpointVolume

def get_volume_range(volume) -> VolumeRange:
    _minimum, _maximum, _ = volume.GetVolumeRange()
    return VolumeRange(minimum=_minimum, maximum=_maximum)

def map_distance_to_volume(
        distance: float,
        mapping_config: VolumeMappingConfig,
        volume_range: VolumeRange
) -> float:
    return float(
        np.interp(
            distance,
            [mapping_config.min_distance, mapping_config.max_distance],
            [volume_range.minimum, volume_range.maximum]
        )
    )

def map_distance_to_bar(
        distance: float,
        mapping_config: VolumeMappingConfig
) -> float:
    return float(
        np.interp(
            distance,
            [mapping_config.min_distance, mapping_config.max_distance],
            [mapping_config.max_bar_y, mapping_config.min_bar_y]
        )
    )

def apply_volume_gesture(
        measurement: DistanceMeasurement,
        state: GestureSessionState,
        ui_state: VolumeUiState,
        controller: PinchModeController,
        mapping_config: VolumeMappingConfig,
        volume,
        volume_range: VolumeRange,
        now: float
) -> None:
    controller.update_toggle(measurement.length, state)
    if not state.volume_mode:
        return

    volume_level = map_distance_to_volume(measurement.length, mapping_config, volume_range)
    ui_state.volume_bar_y = map_distance_to_bar(measurement.length, mapping_config)
    volume.SetMasterVolumeLevel(volume_level, None)

    if controller.should_exit_on_stable_hold(measurement.length, state, now):
        controller.exit_mode(state)

def draw_gesture_feedback(img, measurement: DistanceMeasurement, state: GestureSessionState) -> None:
    cv2.line(
        img,
        (measurement.start.x, measurement.start.y),
        (measurement.end.x, measurement.end.y),
        (255, 0, 255),
        2
    )
    cv2.circle(
        img,
        (measurement.center.x, measurement.center.y),
        10,
        (0, 255, 0) if state.volume_mode else (0, 0, 255),
        cv2.FILLED
    )

def draw_volume_feedback(img, ui_state: VolumeUiState, state: GestureSessionState) -> None:
    cv2.rectangle(img, (50, 150), (85,400), (0, 255, 0), 2)
    cv2.rectangle(img, (50, int(ui_state.volume_bar_y)), (85,400), (0, 255, 0), cv2.FILLED)
    cv2.putText(
        img,
        f"Volume mode: {'ON' if state.volume_mode else 'OFF'}",
        (40, 110),
        cv2.FONT_HERSHEY_PLAIN,
        2,
        (0, 255, 0) if state.volume_mode else (0, 0, 255),
        2
    )

def draw_fps(img, previous_time: float, current_time: float) -> float:
    fps = 1 / (current_time - previous_time) if previous_time else 0
    cv2.putText(
        img,
        f"FPS: {int(fps)}",
        (40,70),
        cv2.FONT_HERSHEY_PLAIN,
        3,
        (255, 0, 0),
        3
    )
    return current_time

def main() -> None:
    camera_config = CameraConfig()
    mapping_config = VolumeMappingConfig()
    toggle_config = PinchToggleConfig()
    exit_config = StableHoldExitConfig()

    gesture_state = GestureSessionState()
    ui_state = VolumeUiState()
    controller = PinchModeController(GestureMode.VOLUME, toggle_config, exit_config)
    previous_time = 0.0

    cap = create_camera(camera_config)
    detector = HandDetector()
    volume = get_system_volume()
    volume_range = get_volume_range(volume)

    while True:
        success, img = cap.read()
        if not success:
            continue

        now = time.time()
        img = detector.find_hands(img)
        landmarks = detector.find_position(img, landmark_ids=[THUMB_TIP_ID, INDEX_TIP_ID])
        if landmarks:
            measurement = measure_distance_between_landmarks(landmarks, THUMB_TIP_ID, INDEX_TIP_ID)
            apply_volume_gesture(measurement, gesture_state, ui_state, controller, mapping_config, volume, volume_range, now)
            draw_gesture_feedback(img, measurement, gesture_state)
        else:
            controller.reset_tracking(gesture_state)

        draw_volume_feedback(img, ui_state, gesture_state)
        previous_time = draw_fps(img, previous_time, now)

        cv2.imshow(WINDOW_NAME, img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()