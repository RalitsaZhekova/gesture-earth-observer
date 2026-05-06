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
    return LandmarkPoint(
        x=landmarks[landmark_id][1],
        y=landmarks[landmark_id][2]
    )

def measure_distance_between_landmarks(
        landmarks: list[list[int]],
        start_landmark_id: int,
        end_landmark_id: int
) -> DistanceMeasurement:
    start = get_landmark_point(landmarks, start_landmark_id)
    end = get_landmark_point(landmarks, end_landmark_id)
    center = LandmarkPoint(
        x=int((start.x + end.x) / 2),
        y=int((start.y + end.y) / 2)
    )
    length = math.hypot(end.x - start.x, end.y - start.y)
    return DistanceMeasurement(start=start, end=end, center=center, length=length)