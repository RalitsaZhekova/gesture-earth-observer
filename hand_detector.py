import cv2
import mediapipe as mp


class HandDetector:
    def __init__(
        self,
        mode: bool = False,
        max_hands: int = 2,
        detection_conf: float = 0.5,
        track_conf: float = 0.5,
    ) -> None:
        self.results = None

        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils

        self.hands = self.mp_hands.Hands(
            static_image_mode=mode,
            max_num_hands=max_hands,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=track_conf,
        )

    def find_hands(self, img, draw: bool = True):
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(rgb_img)

        if draw and self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    img,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                )

        return img

    def find_position(
        self,
        img,
        hand_id: int = 0,
        landmark_ids: list[int] | None = None,
        draw: bool = True,
    ) -> list[list[int]]:
        landmark_ids = landmark_ids or []
        landmarks: list[list[int]] = []

        if not self.results or not self.results.multi_hand_landmarks:
            return landmarks

        if hand_id >= len(self.results.multi_hand_landmarks):
            return landmarks

        tracked_hand = self.results.multi_hand_landmarks[hand_id]
        height, width, _ = img.shape

        for landmark_id, landmark in enumerate(tracked_hand.landmark):
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            landmarks.append([landmark_id, x, y])

            if draw and landmark_id in landmark_ids:
                cv2.circle(img, (x, y), 10, (255, 0, 255), cv2.FILLED)

        return landmarks