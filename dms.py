import argparse
import cv2
import numpy as np
import torch
import tensorflow as tf
import RPi.GPIO as GPIO  # Add GPIO import
import time
import warnings
import pygame.mixer  # Add pygame.mixer import
import os

from dms_utils.dms_utils import load_and_preprocess_image, ACTIONS
from net import MobileNet
from facial_tracking.facialTracking import FacialTracker
import facial_tracking.conf as conf

# GPIO Setup
EYES_CLOSED_PIN = 17  # GPIO pin for eyes closed
YAWN_PIN = 27        # GPIO pin for yawning
MOBILE_PIN = 22      # GPIO pin for mobile phone detection

# Adjust constants for faster response
MIN_SIGNAL_DURATION = 0.5  # Reduced from 2.0 to 0.5 seconds
SIGNAL_RESET_TIME = 0.5    # Reduced from 1.0 to 0.5 seconds

# Add after other constants
AUDIO_DIR = "audio_warnings"  # Directory containing warning sound files
WARNING_SOUND = "warning.wav"  # Single warning sound file

class SignalHandler:
    def __init__(self, pin, use_sound=False):
        self.pin = pin
        self.active_since = None
        self.last_state = False
        self.use_sound = use_sound
        self.last_alert_time = 0
        self.alert_cooldown = 2.0  # Reduced cooldown from 3.0 to 2.0 seconds

    def update(self, new_state):
        if new_state != self.last_state:
            print(f"\n🔄 GPIO {self.pin} state changed to: {'HIGH' if new_state else 'LOW'}")
            
        if new_state and not self.last_state:
            self.active_since = time.time()
            print(f"⏱️ Starting timer for GPIO {self.pin}")
        elif not new_state and self.last_state:
            if self.active_since:
                print(f"⏱️ GPIO {self.pin} signal reset after {time.time() - self.active_since:.1f}s")
            self.active_since = None
            
        self.last_state = new_state
        
        if self.active_since and (time.time() - self.active_since) >= MIN_SIGNAL_DURATION:
            GPIO.output(self.pin, GPIO.HIGH)
            current_time = time.time()
            if self.use_sound and (current_time - self.last_alert_time) >= self.alert_cooldown:
                try:
                    print(f"\n🔊 Playing warning sound for GPIO {self.pin}")
                    pygame.mixer.Sound(WARNING_SOUND).play()
                    self.last_alert_time = current_time
                    print(f"✅ Sound played successfully for GPIO {self.pin}")
                except Exception as e:
                    print(f"❌ Audio playback error for GPIO {self.pin}: {e}")
            print(f"⚡ GPIO {self.pin} triggered (active for {(time.time() - self.active_since):.1f}s)")
        else:
            GPIO.output(self.pin, GPIO.LOW)

def setup_gpio():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(EYES_CLOSED_PIN, GPIO.OUT)
    GPIO.setup(YAWN_PIN, GPIO.OUT)
    GPIO.setup(MOBILE_PIN, GPIO.OUT)
    
    # Initialize all pins to LOW
    GPIO.output(EYES_CLOSED_PIN, GPIO.LOW)
    GPIO.output(YAWN_PIN, GPIO.LOW)
    GPIO.output(MOBILE_PIN, GPIO.LOW)
    
    print("GPIO initialized successfully")

def setup_audio():
    """Initialize audio system"""
    try:
        pygame.mixer.init()
        print("Audio system initialized successfully")
    except Exception as e:
        print(f"Failed to initialize audio: {e}")

# Create signal handlers
eyes_signal = SignalHandler(EYES_CLOSED_PIN, use_sound=True)
yawn_signal = SignalHandler(YAWN_PIN, use_sound=True)
mobile_signal = SignalHandler(MOBILE_PIN, use_sound=True)

def infer_one_frame(image, interpreter, yolo_model, facial_tracker):
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Initialize status variables with default values
    eyes_status = 'unknown'
    yawn_status = 'unknown'
    action = ''

    # Process facial tracking with error handling
    try:
        facial_tracker.process_frame(image)
        if facial_tracker.detected:
            eyes_status = facial_tracker.eyes_status if facial_tracker.eyes_status else 'unknown'
            yawn_status = facial_tracker.yawn_status if facial_tracker.yawn_status else 'unknown'
            print(f"\n👁️ Eyes Status: {eyes_status}")
            print(f"😮 Yawn Status: {yawn_status}")
            print(f"Face Detected: {facial_tracker.detected}")
        else:
            print("\n⚠️ No face detected in frame")
    except Exception as e:
        print(f"\n❌ Facial tracking error: {e}")

    # Phone detection
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    yolo_result = yolo_model(rgb_image)
    phone_detected = len(yolo_result.xyxy[0]) > 0
    
    if phone_detected:
        print("\n📱 Phone detected!")
        for detection in yolo_result.xyxy[0]:
            confidence = detection[4].item()
            print(f"Phone confidence: {confidence:.2f}")

    # Debug prints for all states
    print("\n=== Detection Summary ===")
    print(f"Face Detection: {'Success' if facial_tracker.detected else 'Failed'}")
    print(f"Eyes Status: {eyes_status}")
    print(f"Yawn Status: {yawn_status}")
    print(f"Phone Detected: {phone_detected}")

    # Update signals
    eyes_signal.update(eyes_status == 'eye closed')
    yawn_signal.update(yawn_status == 'yawning')
    mobile_signal.update(phone_detected)

    # Visual feedback
    if facial_tracker.detected:
        # Draw face landmarks or rectangle here if needed
        cv2.putText(image, f'Eyes: {eyes_status}', (30,40), 0, 1.0,
                    (0, 0, 255) if eyes_status == 'eye closed' else (0, 255, 0), 2)
        cv2.putText(image, f'Mouth: {yawn_status}', (30,80), 0, 1.0,
                    (0, 0, 255) if yawn_status == 'yawning' else (0, 255, 0), 2)
    else:
        cv2.putText(image, 'No face detected', (30,40), 0, 1.0, (0, 0, 255), 2)

    if phone_detected:
        cv2.putText(image, 'PHONE DETECTED!', (30,120), 0, 1.0, (0, 0, 255), 3)
        # Draw bounding boxes for phones
        for detection in yolo_result.xyxy[0]:
            x1, y1, x2, y2 = map(int, detection[:4])
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return image

def infer(args):
    try:
        setup_gpio()
        setup_audio()  # Add audio setup
        print("Starting DMS monitoring...")
        
        checkpoint = args.checkpoint
        
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=checkpoint)
        interpreter.allocate_tensors()

        # Modified YOLOv5 loading to avoid CUDA warnings
        yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
        yolo_model.classes = [67]  # phone class
        yolo_model.conf = 0.35     
        yolo_model.iou = 0.45      
        yolo_model.max_det = 2     
        # Force CPU mode
        yolo_model = yolo_model.cpu()
        yolo_model.eval()

        # Disable gradients and set to CPU only mode
        torch.set_grad_enabled(False)
        
        # Add this to suppress CUDA warnings
        warnings.filterwarnings('ignore', category=FutureWarning)
        warnings.filterwarnings('ignore', category=UserWarning)

        image_path = args.image
        video_path = args.video
        cam_id = args.webcam
        save = args.save

        facial_tracker = FacialTracker()

        if image_path:
            image = cv2.imread(image_path)
            image = infer_one_frame(image, interpreter, yolo_model, facial_tracker)
            cv2.imwrite('images/test_inferred.jpg', image)
        
        if video_path or cam_id is not None:
            cap = cv2.VideoCapture(video_path) if video_path else cv2.VideoCapture(cam_id)
            
            if cam_id is not None:
                # Adjust camera settings for better responsiveness
                cap.set(3, 240)  # width
                cap.set(4, 180)  # height
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FPS, 15)  # Increased from 10 to 15 FPS
            
            frame_width = int(cap.get(3))
            frame_height = int(cap.get(4))
            fps = cap.get(cv2.CAP_PROP_FPS)

            if save:
                out = cv2.VideoWriter('videos/output.avi',cv2.VideoWriter_fourcc('M','J','P','G'),
                    fps, (frame_width,frame_height))
            
            while True:
                success, image = cap.read()
                if not success:
                    break

                if cam_id is not None:
                    # Reduce frame skipping for faster response
                    cap.grab()  # Skip only 1 frame instead of 3
                
                image = infer_one_frame(image, interpreter, yolo_model, facial_tracker)
                
                if save:
                    out.write(image)
                else:
                    cv2.imshow('DMS', image)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
            cap.release()
            if save:
                out.release()
            cv2.destroyAllWindows()

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        GPIO.cleanup()
    finally:
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