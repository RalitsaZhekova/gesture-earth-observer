from dataclasses import dataclass
from enum import Enum


class GestureMode(Enum):
    IDLE = "idle"
    VOLUME = "volume"
    NAVIGATION = "navigation"
    ZOOM = "zoom"
    PAN = "pan"
    COMMAND = "command"


class GestureAction(str, Enum):
    NONE = "none"
    SWIPE_LEFT = "swipe_left"
    SWIPE_RIGHT = "swipe_right"
    RESET_VIEW = "reset_view"
    TOGGLE_COMMAND = "toggle_command"


@dataclass(frozen=True)
class PinchToggleConfig:
    enter_distance: int = 35
    release_distance: int = 50
    hold_frames: int = 15


@dataclass(frozen=True)
class StableHoldExitConfig:
    tolerance: int = 12
    hold_seconds: float = 1.5


@dataclass(frozen=True)
class TransitionConfig:
    mode_cooldown_seconds: float = 0.35
    swipe_cooldown_seconds: float = 0.5
    reset_cooldown_seconds: float = 0.75


@dataclass
class GestureSessionState:
    active_mode: GestureMode = GestureMode.IDLE
    previous_mode: GestureMode = GestureMode.IDLE
    pinch_frames: int = 0
    pinch_armed: bool = True
    stable_value: float | None = None
    stable_since: float | None = None
    mode_entered_at: float | None = None
    mode_locked_until: float = 0.0
    pending_action: GestureAction = GestureAction.NONE
    action_locked_until: float = 0.0
    zoom_anchor_distance: float | None = None
    zoom_anchor_level: float | None = None

    @property
    def volume_mode(self) -> bool:
        return self.active_mode == GestureMode.VOLUME

    @property
    def zoom_mode(self) -> bool:
        return self.active_mode == GestureMode.ZOOM

    @property
    def command_mode(self) -> bool:
        return self.active_mode == GestureMode.COMMAND

    @property
    def viewer_mode(self) -> bool:
        return self.active_mode in {
            GestureMode.NAVIGATION,
            GestureMode.ZOOM,
            GestureMode.PAN
        }


class GestureStateMachine:
    def __init__(self, transition_config: TransitionConfig | None = None) -> None:
        self.transition_config = transition_config or TransitionConfig()

    def initialize_viewer(self, state: GestureSessionState, now: float) -> None:
        if state.active_mode == GestureMode.IDLE:
            self.enter_mode(state, GestureMode.NAVIGATION, now)

    @staticmethod
    def can_enter_mode(state: GestureSessionState, now: float) -> bool:
        return now >= state.mode_locked_until

    def enter_mode(
        self,
        state: GestureSessionState,
        mode: GestureMode,
        now: float,
        cooldown_seconds: float | None = None
    ) -> bool:
        if state.active_mode == mode:
            return False
        if not self.can_enter_mode(state, now):
            return False

        state.previous_mode = state.active_mode
        state.active_mode = mode
        state.mode_entered_at = now
        state.mode_locked_until = now + (cooldown_seconds or self.transition_config.mode_cooldown_seconds)
        self.clear_transient_tracking(state)
        return True

    def exit_to_navigation(self, state: GestureSessionState, now: float) -> bool:
        return self.enter_mode(state, GestureMode.NAVIGATION, now)

    def reset_to_idle(self, state: GestureSessionState, now: float) -> bool:
        return self.enter_mode(state, GestureMode.IDLE, now)

    def enter_zoom(self, state: GestureSessionState, now: float) -> bool:
        return self.enter_mode(state, GestureMode.ZOOM, now)

    def enter_pan(self, state: GestureSessionState, now: float) -> bool:
        return self.enter_mode(state, GestureMode.PAN, now)

    def toggle_command_mode(self, state: GestureSessionState, now: float) -> bool:
        if state.active_mode == GestureMode.COMMAND:
            return self.exit_to_navigation(state, now)
        return self.enter_mode(state, GestureMode.COMMAND, now)

    @staticmethod
    def can_emit_action(state: GestureSessionState, now: float) -> bool:
        return now >= state.action_locked_until

    def emit_action(
        self,
        state: GestureSessionState,
        action: GestureAction,
        now: float,
        cooldown_seconds: float | None = None
    ) -> bool:
        if not self.can_emit_action(state, now):
            return False

        state.pending_action = action
        state.action_locked_until = now + (cooldown_seconds or self.transition_config.swipe_cooldown_seconds)
        return True

    @staticmethod
    def consume_action(state: GestureSessionState) -> GestureAction:
        action = state.pending_action
        state.pending_action = GestureAction.NONE
        return action

    def request_swipe_left(self, state: GestureSessionState, now: float) -> bool:
        if state.active_mode != GestureMode.NAVIGATION:
            return False
        return self.emit_action(state, GestureAction.SWIPE_LEFT, now)

    def request_swipe_right(self, state: GestureSessionState, now: float) -> bool:
        if state.active_mode != GestureMode.NAVIGATION:
            return False
        return self.emit_action(state, GestureAction.SWIPE_RIGHT, now)

    def request_reset_view(self, state: GestureSessionState, now: float) -> bool:
        emitted = self.emit_action(
            state,
            GestureAction.RESET_VIEW,
            now,
            cooldown_seconds=self.transition_config.reset_cooldown_seconds
        )
        if emitted:
            self.exit_to_navigation(state, now)
        return emitted

    @staticmethod
    def clear_transient_tracking(state: GestureSessionState) -> None:
        state.pinch_frames = 0
        state.pinch_armed = True
        state.stable_value = None
        state.stable_since = None


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

    @staticmethod
    def clear_stable_hold(state: GestureSessionState) -> None:
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

        if state.stable_value is None:
            state.stable_value = value
            state.stable_since = now
            return False

        if abs(value - state.stable_value) <= self.exit_config.tolerance:
            return now - state.stable_since >= self.exit_config.hold_seconds

        state.stable_value = value
        state.stable_since = now
        return False