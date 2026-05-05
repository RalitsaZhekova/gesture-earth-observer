import cv2
import mediapipe as mp
import time

from hand_detector import HandDetector

def main():
    p_time = 0
    c_time = 0

    cap = cv2.VideoCapture(0)
    detector = HandDetector()

    while True:
        success, img = cap.read()
        img = detector.find_hands(img)
        lm_list = detector.find_position(img)

        if len(lm_list) > 0:
            print(lm_list[8])

        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time
        cv2.putText(
            img,
            "FPS: {:.2f}".format(fps),
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            3
        )

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()