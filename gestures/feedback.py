import cv2
import numpy as np

from gesture_state import GestureMode, GestureSessionState
from hand_geometry import DistanceMeasurement

def draw_gesture_feedback(
        frame: np.ndarray,
        measurement: DistanceMeasurement | None,
        gesture_state: GestureSessionState
) -> None:
    if measurement is None:
        return

    if gesture_state.active_mode == GestureMode.ZOOM:
        color = (0, 255, 0)
    elif gesture_state.active_mode == GestureMode.PAN:
        color = (0, 255, 255)
    else:
        color = (0, 0, 255)

    cv2.line(
        frame,
        (measurement.start.x, measurement.start.y),
        (measurement.end.x, measurement.end.y),
        (255, 0, 255),
        2
    )
    cv2.circle(
        frame,
        (measurement.center.x, measurement.center.y),
        10,
        color,
        cv2.FILLED
    )

def draw_ui(
    frame: np.ndarray,
    gesture_state: GestureSessionState,
    map_snapshot: dict[str, float | str],
    previous_time: float,
    current_time: float,
) -> float:
    fps = 1 / (current_time - previous_time) if previous_time else 0
    zoom_level = float(map_snapshot["zoom_level"])
    center_lat = float(map_snapshot["center_lat"])
    center_lng = float(map_snapshot["center_lng"])
    cv2.putText(
        frame,
        f"Mode: {gesture_state.active_mode.value.upper()}",
        (30, 45),
        cv2.FONT_HERSHEY_PLAIN,
        2,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        frame,
        f"Map zoom: {zoom_level:.1f}",
        (30, 80),
        cv2.FONT_HERSHEY_PLAIN,
        2,
        (0, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        "Open palm: swipe satellites. Thumb+index: zoom. Index+middle closed: pan.",
        (30, 115),
        cv2.FONT_HERSHEY_PLAIN,
        0.85,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        f"Center: {center_lat:.4f}, {center_lng:.4f}",
        (30, 145),
        cv2.FONT_HERSHEY_PLAIN,
        1.2,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        "Imagery map opens in your browser at http://127.0.0.1:8765",
        (30, 175),
        cv2.FONT_HERSHEY_PLAIN,
        1.0,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (30, 210),
        cv2.FONT_HERSHEY_PLAIN,
        2,
        (255, 0, 0),
        2,
    )
    return current_time