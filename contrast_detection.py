import argparse
import numpy as np
from skimage.exposure import is_low_contrast
import cv2

class ContrastDetection:
    def __init__(self):
        self.contrast_text = None
        self.edge_perecentage = 0

    def detect_contrast(self, frame, contrast_percentage):
        ap = argparse.ArgumentParser()
        ap.add_argument("-i", "--input", type=str, default="", help="optional path to video file")
        ap.add_argument("-t", "--thresh", type=float, default=0.35, help="threshold for low contrast")
        args = vars(ap.parse_args())

        print("[INFO] accessing video stream...")

        contrast_percentage = float(contrast_percentage)
        edged = None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if is_low_contrast(gray, fraction_threshold=args["thresh"]):
            self.contrast_text = "Low contrast"
        else:
            self.contrast_text = "Better contrast"

            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edged = cv2.Canny(blurred, 30, 150)

            # Calculate edge percentage
            edge_pixels = cv2.countNonZero(edged)
            total_pixels = edged.size
            edge_percentage = (edge_pixels / total_pixels) * 100

            # Determine lighting condition
            if edge_percentage < contrast_percentage:
                print(f"comparing with respect to {contrast_percentage}")
                light_condition = "Bad Lighting"
            else:
                light_condition = "Good Lighting" 

            # Append lighting condition to text
            self.contrast_text += f" with {light_condition} conditions" 
            self.edge_perecentage = (edge_percentage)*100
