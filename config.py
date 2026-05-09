from dataclasses import dataclass


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
class PanGestureConfig:
    enter_distance: int = 28
    release_distance: int = 42
    hold_frames: int = 12
    clutch_distance: int = 48
    pixels_to_longitude_scale: float = 5
    smoothing_factor: float = 0.3
    movement_deadband_pixels: float = 2.5


@dataclass(frozen=True)
class SwipeGestureConfig:
    entry_frames: int = 5
    min_travel_pixels: float = 110.0
    max_vertical_drift_pixels: float = 180.0
    timeout_seconds: float = 2.0
    lost_palm_grace_seconds: float = 0.55


@dataclass(frozen=True)
class SatellitePreset:
    name: str
    caption: str
    center_lat: float
    center_lng: float
    zoom_level: float


SATELLITE_PRESETS = [
    SatellitePreset(
        name="Atlantic Arc",
        caption="A wide atmospheric pass between the Americas, Europe, and Africa.",
        center_lat=20.0,
        center_lng=-35.0,
        zoom_level=3.2,
    ),
    SatellitePreset(
        name="Sahara Bloom",
        caption="North Africa with Mediterranean edge light and desert texture.",
        center_lat=24.0,
        center_lng=14.0,
        zoom_level=4.0,
    ),
    SatellitePreset(
        name="Himalayan Spine",
        caption="Snow structure and terrain drama across Central and South Asia.",
        center_lat=29.5,
        center_lng=86.5,
        zoom_level=5.0,
    ),
    SatellitePreset(
        name="Austral Tides",
        caption="Australia and the Coral Sea framed as a clean southern sweep.",
        center_lat=-23.0,
        center_lng=136.0,
        zoom_level=4.1,
    ),
    SatellitePreset(
        name="Patagonia Edge",
        caption="South America tapering into ice, cloud, and ocean contrast.",
        center_lat=-43.5,
        center_lng=-71.0,
        zoom_level=4.5,
    ),
]


@dataclass(frozen=True)
class MapServerConfig:
    host: str = "127.0.0.1"
    port: int = 8765
    auto_open_browser: bool = True