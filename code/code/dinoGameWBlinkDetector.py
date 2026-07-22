import os

import cv2
import dlib
import numpy as np
import pyautogui
import time

from utils import eye_aspect_ratio

# The shape predictor lives in the repo root, two directories above this script.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def jump():
    pyautogui.keyDown('space')
    time.sleep(0.01)
    print("jump")
    pyautogui.keyUp('space')


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor(os.path.join(REPO_ROOT, "shape_predictor_68_face_landmarks.dat"))

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: failed to read frame from webcam.")
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = hog_face_detector(gray)
    for face in faces:

        face_landmarks = dlib_facelandmark(gray, face)
        leftEye = []
        rightEye = []

        for n in range(36,42):
        	x = face_landmarks.part(n).x
        	y = face_landmarks.part(n).y
        	leftEye.append((x,y))
        	next_point = n+1
        	if n == 41:
        		next_point = 36
        	x2 = face_landmarks.part(next_point).x
        	y2 = face_landmarks.part(next_point).y
        	cv2.line(frame,(x,y),(x2,y2),(0,255,0),1)

        for n in range(42,48):
        	x = face_landmarks.part(n).x
        	y = face_landmarks.part(n).y
        	rightEye.append((x,y))
        	next_point = n+1
        	if n == 47:
        		next_point = 42
        	x2 = face_landmarks.part(next_point).x
        	y2 = face_landmarks.part(next_point).y
        	cv2.line(frame,(x,y),(x2,y2),(0,255,0),1)

        left_ear = eye_aspect_ratio(np.array(leftEye))
        right_ear = eye_aspect_ratio(np.array(rightEye))

        EAR = (left_ear+right_ear)/2
        EAR = round(EAR,2)
       # print(EAR)
        if EAR<0.22:
            jump()
            # cv2.putText(frame,"JUMP",(20,100), cv2.FONT_HERSHEY_SIMPLEX,3,(0,0,255),4)
            # cv2.imwrite("image.jpg",frame)

    cv2.imshow("Dino Game with Blink Detection", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()