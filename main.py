from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label 
from kivy.clock import Clock
from kivy.graphics.texture import Texture

from blink_detection import blink_detection
from contrast_detection import ContrastDetection
from distance_measurement import DistanceMeasurement
from drowsiness_detection import yawn_detection

import cv2
import threading 

from kivy.uix.popup import Popup
from kivy.uix.gridlayout import GridLayout
from kivy.uix.textinput import TextInput

class SettingsPopup(Popup): 
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.title = 'Settings'
        self.size_hint = (None, None)
        self.size = (500, 200)

        self.layout = GridLayout(cols=2, spacing=10, padding=10)

        self.contrast = TextInput(text='0.35', input_type='number', input_filter='float', multiline=False)

        self.layout.add_widget(Label(text='Contrast:'))
        self.layout.add_widget(Label(text='EAR:')) 
        self.layout.add_widget(self.contrast)

        self.save_button = Button(text='Save', size_hint=(1, None), height=40) 
        self.save_button.bind(on_press=self.save_settings) 
        self.layout.add_widget(self.save_button)

        self.add_widget(self.layout)

    def save_settings(self, instance):
        self.dismiss()

class BuddyApp(App):
     
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.capture = cv2.VideoCapture(0)
        self.settings_popup = SettingsPopup()
        self.current_frame = None
        self.update_event = None
        self.frame_lock = threading.Lock()
        self.frame_update_thread = threading.Thread(target=self.update_frame, daemon=True) 
        self.blink_detection = blink_detection()
        self.contrast_detection = ContrastDetection()
        self.distance_measurement = DistanceMeasurement()
        self.drowsiness_detection = yawn_detection()

    def build(self):
        self.cam = Image(size_hint=(1, .8))
        self.button = Button(text="Start monitoring", size_hint=(1, .1))

        self.blinkRate = Label(text ="Blink Rate: ", size_hint=(1, .1))
        self.drowsiness = Label(text ="Drowsiness : ", size_hint=(1, .1))
        self.contrast = Label(text ="Contrastness: ", size_hint=(1, .1)) 
        self.distance = Label(text ="Contrastness: ", size_hint=(1, .1)) 

        self.settings_button = Button(text="Settings", size_hint=(None, None), size=(100, 80), pos=(10, 10))
        self.settings_button.bind(on_press=self.open_settings)

        self.button.bind(on_press=self.start_monitoring) 

        layout = BoxLayout(orientation='vertical')

        layout.add_widget(self.cam)
        layout.add_widget(self.button)
        layout.add_widget(self.settings_button) 
        layout.add_widget(self.blinkRate)
        layout.add_widget(self.drowsiness)
        layout.add_widget(self.contrast)
        layout.add_widget(self.distance)

        return layout
    
    def open_settings(self, instance):
        self.settings_popup.open()

    def start_monitoring(self, instance):
        self.frame_update_thread.start()
        self.update_event = Clock.schedule_interval(self.process_frame, 1.0 / 10.0)  # 10frames per second

    def update_frame(self): 
        while True:
            ret, frame = self.capture.read() 
            if ret:
                with self.frame_lock: 
                    self.current_frame = frame 

    def process_frame(self, dt): 

        with self.frame_lock:
            frame = self.current_frame

        if frame is not None:        
            self.display_gui(frame)

            self.blink_detection.process_blink_detection(frame)
            contrast =float(self.settings_popup.contrast.text)
            self.drowsiness_detection.process_yawn_detection(frame)
            self.contrast_detection.detect_contrast(frame, contrast)
            self.distance_measurement.detect_distance(frame)
            self.update_gui()
    
    def display_gui(self,frame):
        frame = cv2.flip(frame, 0)
        buf = frame.tobytes()  
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.cam.texture = img_texture
    
    def update_gui(self):
        self.blinkRate.text = f"Blink Rate: {self.blink_detection.blink_counter} {self.blink_detection.blinkRate_response_text}"
        self.drowsiness.text = f"Drowsiness: {self.drowsiness_detection.yawn_response_text}"
        self.contrast.text = f"Contrastness: {self.contrast_detection.contrast_text} {self.contrast_detection.edge_perecentage}"
        self.distance.text = f"Distance: {self.distance_measurement.distance_value}" 

if __name__ == "__main__":
    BuddyApp().run() 