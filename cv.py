import cv2
import mediapipe as mp
import math
from parser import HandParser
import pyaudio

cam = cv2.VideoCapture(0)
mp_draw = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
finger_index = [4,8]

with mp_hands.Hands(min_detection_confidence = 0.8, min_tracking_confidence = 0.5) as hands:
    while True:
        ret,frame = cam.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray.flags.writeable = False
        results = hands.process(gray)
        gray = cv2.cvtColor(gray, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmark in results.multi_hand_landmarks:

                parsed_points = HandParser(hand_landmark)

                points = parsed_points.getPointAtFinger(finger_index, frame)

                wrist = hand_landmark.landmark[0]
                root = hand_landmark.landmark[5]

                distance = math.dist((wrist.x,wrist.y), (root.x,root.y))

                for p in points:
                    cv2.circle(gray, (p[0],p[1]), radius=10,color=(0,255,255), thickness =2)
                    
                cv2.line(gray, points[0],  points[1], (0,255,255),thickness=2)


                mp_draw.draw_landmarks(gray, hand_landmark, mp_hands.HAND_CONNECTIONS)



        cv2.imshow('frame', gray)
        if cv2.waitKey(1) == ord('a'):
            break

cv2.destroyAllWindows()
cam.release()