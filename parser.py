import cv2
import mediapipe as mp


class HandParser():
    def __init__(self,landmark):
        self.landmark = landmark


    def getPointAtFinger(self, finger_index, frame):
        points = []
        for i in finger_index:
            point = self.landmark.landmark[i]

            h,w, _ = frame.shape
            x_px = int(point.x * w)
            y_px = int(point.y * h)
            points.append([x_px,y_px])

        return points

