from dataclasses import dataclass
import math


@dataclass(frozen=True)
class LandmarkPoint:
    x: int
    y: int


@dataclass(frozen=True)
class DistanceMeasurement:
    start: LandmarkPoint
    end: LandmarkPoint
    center: LandmarkPoint
    length: float


def get_landmark_point(landmarks: list[list[int]], landmark_id: int) -> LandmarkPoint:
    _, x, y = landmarks[landmark_id]
    return LandmarkPoint(x=x, y=y)


def measure_distance_between_landmarks(
    landmarks: list[list[int]],
    start_landmark_id: int,
    end_landmark_id: int,
) -> DistanceMeasurement:
    start = get_landmark_point(landmarks, start_landmark_id)
    end = get_landmark_point(landmarks, end_landmark_id)

    center = LandmarkPoint(
        x=(start.x + end.x) // 2,
        y=(start.y + end.y) // 2,
    )

    return DistanceMeasurement(
        start=start,
        end=end,
        center=center,
        length=math.hypot(end.x - start.x, end.y - start.y),
    )