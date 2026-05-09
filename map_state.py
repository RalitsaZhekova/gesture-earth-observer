import threading
import time
from dataclasses import dataclass

from config import SATELLITE_PRESETS
from gesture_state import GestureSessionState, GestureStateMachine, GestureAction


@dataclass
class MapViewState:
    zoom_level: float = SATELLITE_PRESETS[0].zoom_level
    target_zoom_level: float = SATELLITE_PRESETS[0].zoom_level
    center_lat: float = SATELLITE_PRESETS[0].center_lat
    center_lng: float = SATELLITE_PRESETS[0].center_lng
    active_mode: str = "idle"
    satellite_index: int = 0
    satellite_name: str = SATELLITE_PRESETS[0].name
    satellite_caption: str = SATELLITE_PRESETS[0].caption
    transition_direction: str = "none"
    transition_nonce: int = 0
    updated_at: float = 0.0

    def snapshot(self) -> dict[str, float | str]:
        return {
            "zoom_level": self.zoom_level,
            "target_zoom_level": self.target_zoom_level,
            "center_lat": self.center_lat,
            "center_lng": self.center_lng,
            "active_mode": self.active_mode,
            "satellite_index": self.satellite_index,
            "satellite_count": len(SATELLITE_PRESETS),
            "satellite_name": self.satellite_name,
            "satellite_caption": self.satellite_caption,
            "transition_direction": self.transition_direction,
            "transition_nonce": self.transition_nonce,
            "updated_at": self.updated_at
        }


class SharedMapState:
    def __init__(self) -> None:
        self._state = MapViewState()
        self._lock = threading.Lock()

    def update(self, **changes) -> None:
        with self._lock:
            updated_at = changes.pop("updated_at", None)

            for field_name, value in changes.items():
                if value is not None:
                    setattr(self._state, field_name, value)

            self._state.updated_at = updated_at if updated_at is not None else time.time()

    def snapshot(self) -> dict[str, float | str]:
        with self._lock:
            return dict(self._state.snapshot())

def apply_pending_map_action(
    gesture_state: GestureSessionState,
    state_machine: GestureStateMachine,
    shared_map_state: SharedMapState,
    now: float
) -> None:
    action = state_machine.consume_action(gesture_state)
    if action == GestureAction.NONE:
        return

    snapshot = shared_map_state.snapshot()
    satellite_index = int(snapshot["satellite_index"])
    transition_nonce = int(snapshot["transition_nonce"])
    if action == GestureAction.SWIPE_LEFT:
        satellite_index = (satellite_index - 1) % len(SATELLITE_PRESETS)
        transition_direction = "left"
    elif action == GestureAction.SWIPE_RIGHT:
        satellite_index = (satellite_index + 1) % len(SATELLITE_PRESETS)
        transition_direction = "right"
    else:
        return

    preset = SATELLITE_PRESETS[satellite_index]
    shared_map_state.update(
        zoom_level=preset.zoom_level,
        target_zoom_level=preset.zoom_level,
        center_lat=preset.center_lat,
        center_lng=preset.center_lng,
        active_mode=gesture_state.active_mode.value,
        satellite_index=satellite_index,
        satellite_name=preset.name,
        satellite_caption=preset.caption,
        transition_direction=transition_direction,
        transition_nonce=transition_nonce + 1,
        updated_at=now
    )