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
    def __init__(self, pin, use_sound=False):  # Modified to use boolean flag
        self.pin = pin
        self.active_since = None
        self.last_state = False
        self.use_sound = use_sound  # Whether this signal should trigger sound
        self.last_alert_time = 0
        self.alert_cooldown = 3.0  # Seconds between audio alerts

    def update(self, new_state):
        if new_state != self.last_state:
            print(f"GPIO {self.pin} state changed to: {'HIGH' if new_state else 'LOW'}")
            
        if new_state and not self.last_state:
            self.active_since = time.time()
        elif not new_state and self.last_state:
            print(f"GPIO {self.pin} signal reset after {time.time() - self.active_since:.1f}s")
            self.active_since = None
            
        self.last_state = new_state
        
        if self.active_since and (time.time() - self.active_since) >= MIN_SIGNAL_DURATION:
            GPIO.output(self.pin, GPIO.HIGH)
            # Play sound if cooldown has elapsed and sound is enabled
            current_time = time.time()
            if self.use_sound and (current_time - self.last_alert_time) >= self.alert_cooldown:
                try:
                    print(f"\n🔊 Playing warning sound for GPIO {self.pin}")  # Added debug print
                    pygame.mixer.Sound(WARNING_SOUND).play()
                    self.last_alert_time = current_time
                    print(f"✅ Sound played successfully")  # Added success confirmation
                except Exception as e:
                    print(f"❌ Audio playback error: {e}")  # Added error emoji
            print(f"GPIO {self.pin} triggered (active for {(time.time() - self.active_since):.1f}s)")
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
    
    eyes_status = ''
    yawn_status = ''
    action = ''

    facial_tracker.process_frame(image)
    if facial_tracker.detected:
        eyes_status = facial_tracker.eyes_status
        yawn_status = facial_tracker.yawn_status

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    yolo_result = yolo_model(rgb_image)
    phone_detected = len(yolo_result.xyxy[0]) > 0  # Flag for phone detection

    # Draw bounding boxes for detected phones
    if phone_detected:
        for detection in yolo_result.xyxy[0]:
            if len(detection) >= 6:  # Ensure we have all required values
                x1, y1, x2, y2, confidence, class_id = detection.cpu().numpy()
                # Draw rectangle around phone
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                # Add confidence score
                conf_text = f'Phone: {confidence:.2f}'
                cv2.putText(image, conf_text, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (0, 0, 255), 2)

    # Prepare input data for TFLite
    rgb_image = cv2.resize(rgb_image, (224,224))
    rgb_image = tf.expand_dims(rgb_image, 0)
    rgb_image = tf.cast(rgb_image, tf.float32)

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], rgb_image)
    
    # Run inference
    interpreter.invoke()
    
    # Get output tensor
    y = interpreter.get_tensor(output_details[0]['index'])
    result = np.argmax(y, axis=1)

    # Add debug prints for detections
    print("\n--- Detection Status ---")
    print(f"Eyes Status: {eyes_status}")
    print(f"Yawn Status: {yawn_status}")
    print(f"Phone Detected: {phone_detected}")

    # Update signals with current states and add debug prints
    print("\n--- Signal Updates ---")
    print("Checking eyes closed signal...")
    eyes_signal.update(eyes_status == 'eye closed')
    
    print("Checking yawn signal...")
    yawn_signal.update(yawn_status == 'yawning')
    
    print("Checking mobile signal...")
    mobile_signal.update(result[0] == 0 and phone_detected)

    # Update the action detection logic
    action = ''
    if result[0] == 0 and phone_detected:
        action = "Mobile Phone Detected!"
    elif eyes_status == 'eye closed':
        action = "Warning: Eyes Closed!"
    elif yawn_status == 'yawning':
        action = "Warning: Yawning Detected!"

    # Update text display with more visible colors and clearer messages
    cv2.putText(image, f'Driver eyes: {eyes_status}', (30,40), 0, 1.0,
                (0, 0, 255) if eyes_status == 'eye closed' else (0, 255, 0), 2, lineType=cv2.LINE_AA)
    cv2.putText(image, f'Driver mouth: {yawn_status}', (30,80), 0, 1.0,
                (0, 0, 255) if yawn_status == 'yawning' else (0, 255, 0), 2, lineType=cv2.LINE_AA)
    
    # Simplified phone detection logic
    mobile_status = "MOBILE PHONE DETECTED!" if phone_detected else "No Mobile Phone"
    cv2.putText(image, f'Mobile Status: {mobile_status}', (30,120), 0, 1.0,
                (0, 0, 255) if phone_detected else (0, 255, 0), 3, lineType=cv2.LINE_AA)
    
    # Add debug information
    if phone_detected:
        debug_info = f"Detections: {len(yolo_result.xyxy[0])}"
        cv2.putText(image, debug_info, (30,200), 0, 1.0, (255, 0, 0), 2, lineType=cv2.LINE_AA)
    
    if action:  # Display additional warning if needed
        cv2.putText(image, action, (30,160), 0, 1.0,  # Increased size
                    (0, 0, 255), 3, lineType=cv2.LINE_AA)  # Thicker text
    
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