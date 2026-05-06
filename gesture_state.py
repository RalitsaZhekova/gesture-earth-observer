from dataclasses import dataclass
from enum import Enum


class GestureMode(Enum):
    IDLE = "idle"
    VOLUME = "volume"


@dataclass(frozen=True)
class PinchToggleConfig:
    enter_distance: int = 35
    release_distance: int = 50
    hold_frames: int = 15


@dataclass(frozen=True)
class StableHoldExitConfig:
    tolerance: int = 12
    hold_seconds: float = 1.5


@dataclass
class GestureSessionState:
    active_mode: GestureMode = GestureMode.IDLE
    pinch_frames: int = 0
    pinch_armed: bool = True
    stable_value: float | None = None
    stable_since: float | None = None

    @property
    def volume_mode(self) -> bool:
        return self.active_mode == GestureMode.VOLUME


class PinchModeController:
    def __init__(
            self,
            mode: GestureMode,
            toggle_config: PinchToggleConfig,
            exit_config: StableHoldExitConfig
    ) -> None:
        self.mode = mode
        self.toggle_config = toggle_config
        self.exit_config = exit_config

    def reset_tracking(self, state: GestureSessionState) -> None:
        state.pinch_frames = 0
        state.pinch_armed = True
        self.clear_stable_hold(state)

    def clear_stable_hold(self, state: GestureSessionState) -> None:
        state.stable_value = None
        state.stable_since = None

    def update_toggle(self, value: float, state: GestureSessionState) -> None:
        if value <= self.toggle_config.enter_distance:
            state.pinch_frames += 1
            if state.pinch_armed and state.pinch_frames >= self.toggle_config.hold_frames:
                self.toggle_mode(state)
                state.pinch_armed = False
        elif value >= self.toggle_config.release_distance:
            state.pinch_frames = 0
            state.pinch_armed = True

    def toggle_mode(self, state: GestureSessionState) -> None:
        if state.active_mode == self.mode:
            self.exit_mode(state)
            return

        state.active_mode = self.mode
        self.clear_stable_hold(state)

    def exit_mode(self, state: GestureSessionState) -> None:
        state.active_mode = GestureMode.IDLE
        self.clear_stable_hold(state)

    def should_exit_on_stable_hold(
            self,
            value: float,
            state: GestureSessionState,
            now: float
    ) -> bool:
        if state.active_mode != self.mode:
            self.clear_stable_hold(state)
            return False

        if state.stable_since is None:
            state.stable_value = value
            state.stable_since = now
            return False

        if abs(value - state.stable_value) <= self.exit_config.tolerance:
            return now - state.stable_since >= self.exit_config.hold_seconds

        state.stable_value = value
        state.stable_since = now
        return False