import cv2
import dlib
import time
from scipy.spatial import distance as dist 
from blink_detection import blink_detection 

class yawn_detection:
    def __init__(self):
        self.blink_detection = blink_detection()
        self.yawn_response_text = 'NA'

    def mouth_aspect_ratio(self, mouth):
        A = dist.euclidean(mouth[2], mouth[10])  # 51, 59 
        B = dist.euclidean(mouth[4], mouth[8])   # 53, 57

        C = dist.euclidean(mouth[0], mouth[6])   # 49, 55
        mar = (A + B) / (2.0 * C)
        return mar

    def process_yawn_detection(self, frame):
        MAR_THRESHOLD = 0.6  
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        faces = self.blink_detection.face_detector(gray)

        for face in faces: 
            landmarks = self.blink_detection.predictor(gray, face)
            mouth = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(48, 68)]
            mar = self.mouth_aspect_ratio(mouth)

            if mar > MAR_THRESHOLD:
                self.yawn_response_text = 'yes' 

        return frame
