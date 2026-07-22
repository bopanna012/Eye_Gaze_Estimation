import os
import urllib.request

import cv2
import mediapipe as mp

# MediaPipe's legacy `solutions.holistic` API (used by earlier versions of
# this script) was removed from the package. This uses the current Tasks API
# (HolisticLandmarker), which needs a one-time model bundle download.
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "holistic_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/"
    "holistic_landmarker/float16/1/holistic_landmarker.task"
)

vision = mp.tasks.vision
BaseOptions = mp.tasks.BaseOptions

FACE_CONNECTIONS = vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION
POSE_CONNECTIONS = vision.PoseLandmarksConnections.POSE_LANDMARKS
HAND_CONNECTIONS = vision.HandLandmarksConnections.HAND_CONNECTIONS

FACE_SPEC = vision.drawing_utils.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1)
FACE_CONN_SPEC = vision.drawing_utils.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
RIGHT_HAND_SPEC = vision.drawing_utils.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4)
RIGHT_HAND_CONN_SPEC = vision.drawing_utils.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
LEFT_HAND_SPEC = vision.drawing_utils.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4)
LEFT_HAND_CONN_SPEC = vision.drawing_utils.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
POSE_SPEC = vision.drawing_utils.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4)
POSE_CONN_SPEC = vision.drawing_utils.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)


def ensure_model_downloaded():
    if not os.path.exists(MODEL_PATH):
        print("Downloading holistic landmarker model (one-time, ~13MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def main():
    ensure_model_downloaded()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open webcam.")
        return

    options = vision.HolisticLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
    )

    with vision.HolisticLandmarker.create_from_options(options) as holistic:
        timestamp_ms = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error: failed to read frame from webcam.")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms += 1
            result = holistic.detect_for_video(mp_image, timestamp_ms)

            if result.face_landmarks:
                vision.drawing_utils.draw_landmarks(
                    frame, result.face_landmarks, FACE_CONNECTIONS, FACE_SPEC, FACE_CONN_SPEC
                )
            if result.right_hand_landmarks:
                vision.drawing_utils.draw_landmarks(
                    frame, result.right_hand_landmarks, HAND_CONNECTIONS, RIGHT_HAND_SPEC, RIGHT_HAND_CONN_SPEC
                )
            if result.left_hand_landmarks:
                vision.drawing_utils.draw_landmarks(
                    frame, result.left_hand_landmarks, HAND_CONNECTIONS, LEFT_HAND_SPEC, LEFT_HAND_CONN_SPEC
                )
            if result.pose_landmarks:
                vision.drawing_utils.draw_landmarks(
                    frame, result.pose_landmarks, POSE_CONNECTIONS, POSE_SPEC, POSE_CONN_SPEC
                )

            cv2.imshow("Raw Webcam Feed", frame)

            key = cv2.waitKey(1)
            if key == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
