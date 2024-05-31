import cv2
import dlib 
import time
import datetime 
from scipy.spatial import distance as dist

class blink_detection:
    def __init__(self):
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.face_detector = dlib.get_frontal_face_detector() 
        self.blink_counter = 0
        self.blinkRate_response_text = 'NA'
        self.standard_blinkrate=0 

    def eye_aspect_ratio(self, eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C) 
        return ear
    
    def find_eyes(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector(gray)
        for face in faces:
            landmarks = self.predictor(gray, face)
            left_eye = (landmarks.part(36).x, landmarks.part(36).y)
            right_eye = (landmarks.part(45).x, landmarks.part(45).y)
            return left_eye, right_eye
        return None, None

    def process_blink_detection(self, frame):
        EAR_THRESHOLD = 0.2
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector(gray)
        for face in faces:
            landmarks = self.predictor(gray, face)
            landmarks = [(p.x, p.y) for p in landmarks.parts()]
            left_eye = landmarks[36:42]
            right_eye = landmarks[42:48]
            left_ear = self.eye_aspect_ratio(left_eye)
            right_ear = self.eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0

            if ear < EAR_THRESHOLD:
                self.blink_counter += 1 

    # def is_it_day():
    #     current_time = datetime.datetime.now().time() 
    #     day_start = datetime.time(6, 0, 0)  # 6:00 AM 
    #     day_end = datetime.time(18, 0, 0)   # 6:00 PM 
        
    #     if day_start <= current_time <= day_end:
    #         return True  
    #     return False  

    # def initialize_val(self): 

    #     if(self.is_it_day()):
    #         self.standard_blinkrate = 10 
    #     else
