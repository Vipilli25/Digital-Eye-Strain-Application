import cv2
from blink_detection import blink_detection

class DistanceMeasurement:
    def __init__(self, known_distance=76.2, known_width=14.3):
        # Constants
        self.KNOWN_DISTANCE = known_distance  
        self.KNOWN_WIDTH = known_width 
        self.blink_detection = blink_detection()
        self.distance_value=0
        self.focal_length = 0
    
    def find_focal_length(self, ref_image):
        """Calculate the focal length using a reference image."""
        face_width = self._face_data(ref_image)
        if face_width:  
            self.focal_length = (face_width * self.KNOWN_DISTANCE) / self.KNOWN_WIDTH

    def _face_data(self, image):
        """Detect face using dlib and return face width."""
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.blink_detection.face_detector(gray_image)
        
        for face in faces:
            x = face.left()
            y = face.top()
            w = face.right() - x
            h = face.bottom() - y
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
            return w
        return None


    def find_distance(self, face_width_in_frame):
        """Calculate the distance from the camera to the face."""
        ref_image = cv2.imread("Ref_image.png")
        self.find_focal_length(ref_image)

        if self.focal_length is not None and face_width_in_frame is not None:
            distance = (self.KNOWN_WIDTH * self.focal_length) / face_width_in_frame
            return distance
        return None 

    def detect_distance(self, frame):
        """Detect face in the frame and calculate distance."""
        face_width = self._face_data(frame)
        if face_width:
            distance = self.find_distance(face_width)
            if distance is not None:
                self.distance_value = distance

