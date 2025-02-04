import argparse
import cv2
import numpy as np
import torch
import tensorflow as tf
import RPi.GPIO as GPIO  # Add GPIO import
import time

from dms_utils.dms_utils import load_and_preprocess_image, ACTIONS
from net import MobileNet
from facial_tracking.facialTracking import FacialTracker
import facial_tracking.conf as conf

# GPIO Setup
EYES_CLOSED_PIN = 17  # GPIO pin for eyes closed
YAWN_PIN = 27        # GPIO pin for yawning
MOBILE_PIN = 22      # GPIO pin for mobile phone detection

# Add constants for signal timing
MIN_SIGNAL_DURATION = 2.0  # Minimum duration (seconds) before triggering alert
SIGNAL_RESET_TIME = 1.0    # Time to wait before resetting signal

class SignalHandler:
    def __init__(self, pin):
        self.pin = pin
        self.active_since = None
        self.last_state = False

    def update(self, new_state):
        if new_state and not self.last_state:  # Signal just became active
            self.active_since = time.time()
        elif not new_state and self.last_state:  # Signal just became inactive
            self.active_since = None
        
        self.last_state = new_state
        
        # Only trigger if signal has been active for minimum duration
        if self.active_since and (time.time() - self.active_since) >= MIN_SIGNAL_DURATION:
            GPIO.output(self.pin, GPIO.HIGH)
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

# Create signal handlers
eyes_signal = SignalHandler(EYES_CLOSED_PIN)
yawn_signal = SignalHandler(YAWN_PIN)
mobile_signal = SignalHandler(MOBILE_PIN)

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

    # Update signals with current states
    eyes_signal.update(eyes_status == 'eye closed')
    yawn_signal.update(yawn_status == 'yawning')
    mobile_signal.update(result[0] == 0 and yolo_result.xyxy[0].shape[0] > 0)

    # Control GPIO based on detections
    GPIO.output(EYES_CLOSED_PIN, GPIO.HIGH if eyes_status == 'eye closed' else GPIO.LOW)
    GPIO.output(YAWN_PIN, GPIO.HIGH if yawn_status == 'yawning' else GPIO.LOW)
    GPIO.output(MOBILE_PIN, GPIO.HIGH if (result[0] == 0 and yolo_result.xyxy[0].shape[0] > 0) else GPIO.LOW)

    # Update the action detection logic
    action = ''
    if result[0] == 0 and yolo_result.xyxy[0].shape[0] > 0:
        action = "Mobile Phone Detected!"
    elif eyes_status == 'eye closed':
        action = "Warning: Eyes Closed!"
    elif yawn_status == 'yawning':
        action = "Warning: Yawning Detected!"

    # Update text display with more visible colors and clearer messages
    cv2.putText(image, f'Driver eyes: {eyes_status}', (30,40), 0, 0.7,
                (0, 0, 255) if eyes_status == 'eye closed' else conf.LM_COLOR, 2, lineType=cv2.LINE_AA)
    cv2.putText(image, f'Driver mouth: {yawn_status}', (30,80), 0, 0.7,
                (0, 0, 255) if yawn_status == 'yawning' else conf.CT_COLOR, 2, lineType=cv2.LINE_AA)
    if action:  # Only display action text if there's a warning
        cv2.putText(image, action, (30,120), 0, 0.7,
                    (0, 0, 255), 2, lineType=cv2.LINE_AA)
    
    return image

def infer(args):
    try:
        setup_gpio()
        print("Starting DMS monitoring...")
        
        checkpoint = args.checkpoint
        
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=checkpoint)
        interpreter.allocate_tensors()

        # Replace YOLOv5s with YOLOv5n
        yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5n')
        yolo_model.classes = [67]  # phone class
        # Optional: Set inference size for even faster processing
        yolo_model.conf = 0.25  # Lower confidence threshold for faster inference
        yolo_model.iou = 0.45   # Lower IoU threshold

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
                # Reduce resolution for faster processing
                cap.set(3, 320)  # Reduced width
                cap.set(4, 240)  # Reduced height
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FPS, 15)  # Reduced FPS
            
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
                    # Skip more frames for faster processing
                    cap.grab()  # Skip frame 1
                    cap.grab()  # Skip frame 2
                    cap.grab()  # Skip frame 3
                
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