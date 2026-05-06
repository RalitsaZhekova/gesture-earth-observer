import threading
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import json
import time
import webbrowser

import cv2
import numpy as np

from gesture_state import (
    GestureMode,
    GestureSessionState,
    GestureStateMachine,
    PinchModeController,
    PinchToggleConfig,
    StableHoldExitConfig
)

from hand_detector import HandDetector
from hand_geometry import DistanceMeasurement, measure_distance_between_landmarks

THUMB_TIP_ID = 4
INDEX_TIP_ID = 8
WINDOW_NAME = "Gesture Control"
MAP_TEMPLATE_PATH = Path(__file__).resolve().parent / "web" / "map_view.html"


@dataclass(frozen=True)
class CameraConfig:
    width: int = 640
    height: int = 480


@dataclass(frozen=True)
class ZoomGestureConfig:
    min_distance: int = 35
    max_distance: int = 280
    min_zoom: float = 3.0
    max_zoom: float = 20.0
    zoom_sensitivity: float = 0.05
    smoothing_factor: float = 0.18
    zoom_deadband: float = 0.03


@dataclass(frozen=True)
class MapServerConfig:
    host: str = "127.0.0.1"
    port: int = 8765
    auto_open_browser: bool = True


@dataclass
class MapViewState:
    zoom_level: float = 8.0
    target_zoom_level: float = 8.0
    center_lat: float = 46.8182
    center_lng: float = 8.2275
    active_mode: str = GestureMode.IDLE.value
    updated_at: float = 0.0

    def snapshot(self) -> dict[str, float | str]:
        return {
            "zoom_level": self.zoom_level,
            "target_zoom_level": self.target_zoom_level,
            "center_lat": self.center_lat,
            "center_lng": self.center_lng,
            "active_mode": self.active_mode,
            "updated_at": self.updated_at
        }


class SharedMapState:
    def __init__(self) -> None:
        self._state = MapViewState()
        self._lock = threading.Lock()

    def update(
        self,
        *,
        zoom_level: float | None = None,
        target_zoom_level: float | None = None,
        center_lat: float | None = None,
        center_lng: float | None = None,
        active_mode: str | None = None,
        updated_at: float | None = None
    ) -> None:
        with self._lock:
            if zoom_level is not None:
                self._state.zoom_level = zoom_level
            if target_zoom_level is not None:
                self._state.target_zoom_level = target_zoom_level
            if center_lat is not None:
                self._state.center_lat = center_lat
            if center_lng is not None:
                self._state.center_lng = center_lng
            if active_mode is not None:
                self._state.active_mode = active_mode
            self._state.updated_at = updated_at if updated_at else time.time()

    def snapshot(self) -> dict[str, float | str]:
        with self._lock:
            return dict(self._state.snapshot())


def build_map_handler(shared_state: SharedMapState) -> type[BaseHTTPRequestHandler]:
    class MapRequestHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path in {"/", "/index.html"}:
                self._send_html(MAP_TEMPLATE_PATH.read_text(encoding="utf-8"))
                return

            if self.path == "/state":
                self._send_json(shared_state.snapshot())
                return

            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

        def log_message(self, format: str, *args) -> None:
            return

        def _send_html(self, content: str) -> None:
            payload = content.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _send_json(self, payload: dict[str, float | str]) -> None:
            raw = json.dumps(payload).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    return MapRequestHandler


def start_map_server(config: MapServerConfig, shared_state: SharedMapState) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((config.host, config.port), build_map_handler(shared_state))
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    return server

def open_map_browser(config: MapServerConfig) -> None:
    if not config.auto_open_browser:
        return
    webbrowser.open(f"http://{config.host}:{config.port}", new=1)

def create_camera(camera_config: CameraConfig) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(0)
    cap.set(3, camera_config.width)
    cap.set(4, camera_config.height)
    return cap

def clear_zoom_anchor(gesture_state: GestureSessionState) -> None:
    gesture_state.zoom_anchor_distance = None
    gesture_state.zoom_anchor_level = None

def clamp_zoom_level(zoom_level: float, zoom_config: ZoomGestureConfig) -> float:
    return float(np.clip(zoom_level, zoom_config.min_zoom, zoom_config.max_zoom))

def update_zoom_mode(
    measurement: DistanceMeasurement,
    gesture_state: GestureSessionState,
    zoom_controller: PinchModeController,
    zoom_config: ZoomGestureConfig,
    shared_map_state: SharedMapState,
    now: float
) -> None:
    was_zoom_active = gesture_state.active_mode == GestureMode.ZOOM
    zoom_controller.update_toggle(measurement.length, gesture_state)
    shared_map_state.update(active_mode=gesture_state.active_mode.value, updated_at=now)

    if gesture_state.active_mode != GestureMode.ZOOM:
        if was_zoom_active:
            clear_zoom_anchor(gesture_state)
        return

    snapshot = shared_map_state.snapshot()
    if not was_zoom_active or gesture_state.zoom_anchor_level is None:
        gesture_state.zoom_anchor_distance = measurement.length
        gesture_state.zoom_anchor_level = float(snapshot['zoom_level'])

    snapshot = shared_map_state.snapshot()
    distance_delta = measurement.length - gesture_state.zoom_anchor_distance
    target_zoom = clamp_zoom_level(
        gesture_state.zoom_anchor_level + distance_delta * zoom_config.zoom_sensitivity,
        zoom_config
    )
    current_zoom = float(snapshot['zoom_level'])
    smoothed_zoom = current_zoom + (target_zoom - current_zoom) * zoom_config.smoothing_factor

    if abs(smoothed_zoom - current_zoom) < zoom_config.zoom_deadband:
        smoothed_zoom = current_zoom

    shared_map_state.update(
        zoom_level=smoothed_zoom,
        target_zoom_level=target_zoom,
        active_mode=gesture_state.active_mode.value,
        updated_at=now
    )
    if zoom_controller.should_exit_on_stable_hold(measurement.length, gesture_state, now):
        zoom_controller.exit_mode(gesture_state)
        clear_zoom_anchor(gesture_state)
        shared_map_state.update(active_mode=gesture_state.active_mode.value, updated_at=now)


def draw_gesture_feedback(
    frame: np.ndarray,
    measurement: DistanceMeasurement,
    gesture_state: GestureSessionState
) -> None:
    color = (0, 255, 0) if gesture_state.active_mode == GestureMode.ZOOM else (0, 0, 255)
    cv2.line(
        frame,
        (measurement.start.x, measurement.start.y),
        (measurement.end.x, measurement.end.y),
        (255, 0, 255),
        2
    )
    cv2.circle(frame, (measurement.center.x, measurement.center.y), 10, color, cv2.FILLED)

def draw_ui(
    frame: np.ndarray,
    gesture_state: GestureSessionState,
    zoom_level: float,
    previous_time: float,
    current_time: float
) ->float:
    fps = 1 / (current_time - previous_time) if previous_time else 0

    cv2.putText(
        frame,
        f"Mode: {gesture_state.active_mode.value.upper()}",
        (30, 45),
        cv2.FONT_HERSHEY_PLAIN,
        2,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (30, 180),
        cv2.FONT_HERSHEY_PLAIN,
        2,
        (255, 0, 0),
        2
    )

    return current_time

def main() -> None:
    camera_config = CameraConfig()
    zoom_config = ZoomGestureConfig()
    server_config = MapServerConfig()
    toggle_config = PinchToggleConfig()
    exit_config = StableHoldExitConfig()
    gesture_state = GestureSessionState()
    state_machine = GestureStateMachine()
    zoom_controller = PinchModeController(GestureMode.ZOOM, toggle_config, exit_config)
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
            shared_map_state.update(active_mode=gesture_state.active_mode.value, updated_at=now)

            camera_frame = cv2.flip(camera_frame, 1)
            camera_frame = detector.find_hands(camera_frame)
            landmarks = detector.find_position(camera_frame, landmark_ids=[THUMB_TIP_ID, INDEX_TIP_ID])

            if landmarks:
                measurement = measure_distance_between_landmarks(
                    landmarks,
                    THUMB_TIP_ID,
                    INDEX_TIP_ID
                )
                update_zoom_mode(
                    measurement,
                    gesture_state,
                    zoom_controller,
                    zoom_config,
                    shared_map_state,
                    now
                )
                draw_gesture_feedback(camera_frame, measurement, gesture_state)
            else:
                zoom_controller.reset_tracking(gesture_state)
                shared_map_state.update(active_mode=gesture_state.active_mode.value, updated_at=now)

            zoom_level = float(shared_map_state.snapshot()['zoom_level'])
            previous_time = draw_ui(camera_frame, gesture_state, zoom_level, previous_time, now)

            cv2.imshow(WINDOW_NAME, camera_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        server.shutdown()
        server.server_close()


if __name__ == '__main__':
    main()