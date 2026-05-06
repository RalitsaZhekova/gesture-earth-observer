import cv2
import mediapipe as mp


class HandDetector:
    def __init__(self, mode=False, max_hands=2, detection_conf=0.5, tracking_conf=0.5):
        self.results = None
        self.mode = mode
        self.max_hands = max_hands
        self.detection_conf = detection_conf
        self.tracking_conf = tracking_conf

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_conf,
            min_tracking_confidence=self.tracking_conf
        )

        self.mp_draw = mp.solutions.drawing_utils

    def find_hands(self, img, draw: bool = True) -> str | None:
        if img is None:
            return None

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)

        if self.results.multi_hand_landmarks:
            for hand_lms in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, hand_lms, self.mp_hands.HAND_CONNECTIONS)

        return img

    def find_position(self, img, hand: int = 0, landmark_ids = None, draw: bool = True) -> list[list[int]]:
        if landmark_ids is None:
            landmark_ids = []
        lm_list = []

        if img is None or self.results is None or not self.results.multi_hand_landmarks:
            return lm_list

        if hand < 0 or hand >= len(self.results.multi_hand_landmarks):
            return lm_list

        tracked_hand = self.results.multi_hand_landmarks[hand]

        h, w, _ = img.shape
        for landmark_id, lm in enumerate(tracked_hand.landmark):
            cx, cy = int(lm.x * w), int(lm.y * h)
            lm_list.append([landmark_id, cx, cy])

            if draw and landmark_id in landmark_ids:
                cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

        return lm_list