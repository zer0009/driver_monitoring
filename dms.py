import argparse
import cv2
import numpy as np
import torch
import tensorflow as tf
import time

from dms_utils.dms_utils import load_and_preprocess_image, ACTIONS
from net import MobileNet
from facial_tracking.facialTracking import FacialTracker
import facial_tracking.conf as conf

# Conditional GPIO import
try:
    import RPi.GPIO as GPIO
    IS_RASPBERRY_PI = True
except ImportError:
    IS_RASPBERRY_PI = False
    print("Running on PC - GPIO functionality disabled")

# GPIO Setup
EYES_CLOSED_PIN = 17  # GPIO pin for eyes closed
YAWN_PIN = 27        # GPIO pin for yawning
MOBILE_PIN = 22      # GPIO pin for mobile phone detection

# Add constants for signal timing
MIN_SIGNAL_DURATION = 2.0  # Minimum duration (seconds) before triggering alert
SIGNAL_RESET_TIME = 1.0    # Time to wait before resetting signal

# Enable GPU if available
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Enable OpenCL acceleration for OpenCV
if cv2.ocl.haveOpenCL():
    cv2.ocl.setUseOpenCL(True)

class SignalHandler:
    def __init__(self, pin):
        self.pin = pin
        self.active_since = None
        self.last_state = False

    def update(self, new_state):
        if not IS_RASPBERRY_PI:
            return
            
        if new_state and not self.last_state:
            self.active_since = time.time()
        elif not new_state and self.last_state:
            self.active_since = None
        
        self.last_state = new_state
        
        if self.active_since and (time.time() - self.active_since) >= MIN_SIGNAL_DURATION:
            GPIO.output(self.pin, GPIO.HIGH)
        else:
            GPIO.output(self.pin, GPIO.LOW)

def setup_gpio():
    if not IS_RASPBERRY_PI:
        return
        
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(EYES_CLOSED_PIN, GPIO.OUT)
    GPIO.setup(YAWN_PIN, GPIO.OUT)
    GPIO.setup(MOBILE_PIN, GPIO.OUT)
    
    GPIO.output(EYES_CLOSED_PIN, GPIO.LOW)
    GPIO.output(YAWN_PIN, GPIO.LOW)
    GPIO.output(MOBILE_PIN, GPIO.LOW)
    
    print("GPIO initialized successfully")

# Create signal handlers
eyes_signal = SignalHandler(EYES_CLOSED_PIN)
yawn_signal = SignalHandler(YAWN_PIN)
mobile_signal = SignalHandler(MOBILE_PIN)

def infer_one_frame(image, model, yolo_model, facial_tracker):
    eyes_status = ''
    yawn_status = ''
    action = ''

    facial_tracker.process_frame(image)
    if facial_tracker.detected:
        eyes_status = facial_tracker.eyes_status
        yawn_status = facial_tracker.yawn_status

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    yolo_result = yolo_model(rgb_image)

    rgb_image = cv2.resize(rgb_image, (224,224))
    rgb_image = tf.expand_dims(rgb_image, 0)
    y = model.predict(rgb_image)
    result = np.argmax(y, axis=1)

    # Update signals with current states
    eyes_signal.update(eyes_status == 'eye closed')
    yawn_signal.update(yawn_status == 'yawning')
    mobile_signal.update(result[0] == 0 and yolo_result.xyxy[0].shape[0] > 0)

    # Control GPIO based on detections
    if IS_RASPBERRY_PI:
        GPIO.output(EYES_CLOSED_PIN, GPIO.HIGH if eyes_status == 'eye closed' else GPIO.LOW)
        GPIO.output(YAWN_PIN, GPIO.HIGH if yawn_status == 'yawning' else GPIO.LOW)
        GPIO.output(MOBILE_PIN, GPIO.HIGH if (result[0] == 0 and yolo_result.xyxy[0].shape[0] > 0) else GPIO.LOW)

    if result[0] == 0 and yolo_result.xyxy[0].shape[0] > 0:
        action = list(ACTIONS.keys())[result[0]]
    if result[0] == 1 and eyes_status == 'eye closed':
        action = list(ACTIONS.keys())[result[0]]

    cv2.putText(image, f'Driver eyes: {eyes_status}', (30,40), 0, 1,
                conf.LM_COLOR, 2, lineType=cv2.LINE_AA)
    cv2.putText(image, f'Driver mouth: {yawn_status}', (30,80), 0, 1,
                conf.CT_COLOR, 2, lineType=cv2.LINE_AA)
    cv2.putText(image, f'Driver action: {action}', (30,120), 0, 1,
                conf.WARN_COLOR, 2, lineType=cv2.LINE_AA)
    
    return image

def infer(args):
    try:
        setup_gpio()
        print("Starting DMS monitoring...")
        
        image_path = args.image
        video_path = args.video
        cam_id = args.webcam
        checkpoint = args.checkpoint
        save = args.save

        model = MobileNet()
        model.load_weights(checkpoint)

        yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        yolo_model.classes = [67]

        facial_tracker = FacialTracker()

        if image_path:
            image = cv2.imread(image_path)
            image = infer_one_frame(image, model, yolo_model, facial_tracker)
            cv2.imwrite('images/test_inferred.jpg', image)
        
        if video_path or cam_id is not None:
            cap = cv2.VideoCapture(video_path) if video_path else cv2.VideoCapture(cam_id)
            
            if cam_id is not None:
                # Optimize for Pi camera
                cap.set(3, 320)  # Further reduced width for Pi
                cap.set(4, 240)  # Further reduced height for Pi
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FPS, 15)  # Limit FPS for Pi
            
            frame_width = int(cap.get(3))
            frame_height = int(cap.get(4))
            fps = cap.get(cv2.CAP_PROP_FPS)

            if save:
                out = cv2.VideoWriter('videos/output.avi',
                    cv2.VideoWriter_fourcc(*'MJPG'),
                    fps/4,  # Further reduced output FPS for Pi
                    (frame_width,frame_height))
            
            frame_count = 0
            PROCESS_EVERY_N_FRAMES = 8  # Increased from 4 to 8 for better performance
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_count += 1
                
                if frame_count % PROCESS_EVERY_N_FRAMES != 0:
                    continue
                
                # Reduce resolution even further for Pi
                display_width = 240  # Smaller display size
                display_height = 180
                
                # Use INTER_NEAREST for faster resizing on Pi
                frame = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_NEAREST)
                
                image = infer_one_frame(frame, model, yolo_model, facial_tracker)
                
                if save:
                    out.write(image)
                
                # Optimize display for Pi
                display_frame = cv2.resize(image, (display_width, display_height), 
                                        interpolation=cv2.INTER_NEAREST)
                cv2.namedWindow('DMS', cv2.WINDOW_NORMAL)
                
                if IS_RASPBERRY_PI:
                    cv2.setWindowProperty('DMS', cv2.WND_PROP_FULLSCREEN, 
                                        cv2.WINDOW_FULLSCREEN)
                else:
                    cv2.resizeWindow('DMS', 800, 600)
                    
                cv2.imshow('DMS', display_frame)
                
                # Reduce wait time for better performance
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            cap.release()
            if save:
                out.release()
            cv2.destroyAllWindows()

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        if IS_RASPBERRY_PI:
            GPIO.cleanup()
    finally:
        if IS_RASPBERRY_PI:
            GPIO.cleanup()
            print("GPIO cleaned up")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--image', type=str, default=None, help='Image path')
    p.add_argument('--video', type=str, default=None, help='Video path')
    p.add_argument('--webcam', type=int, default=None, help='Cam ID')
    p.add_argument('--checkpoint', type=str, help='Pre-trained model file path')
    p.add_argument('--save', type=bool, default=False, help='Save video or not')
    args = p.parse_args()

    infer(args)
