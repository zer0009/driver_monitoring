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

    # Resize image before processing to reduce computation
    small_frame = cv2.resize(image, (320, 240))
    
    facial_tracker.process_frame(small_frame)
    if facial_tracker.detected:
        eyes_status = facial_tracker.eyes_status
        yawn_status = facial_tracker.yawn_status

    # Convert to RGB only once
    rgb_image = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    
    # Run YOLO inference on smaller image
    with torch.no_grad():  # Disable gradient calculation
        yolo_result = yolo_model(rgb_image)

    # Prepare input data for TFLite
    tflite_input = cv2.resize(rgb_image, (224, 224))
    tflite_input = tf.expand_dims(tflite_input, 0)
    tflite_input = tf.cast(tflite_input, tf.float32)

    interpreter.set_tensor(input_details[0]['index'], tflite_input)
    interpreter.invoke()
    
    y = interpreter.get_tensor(output_details[0]['index'])
    result = np.argmax(y, axis=1)

    # Update signals with current states
    eyes_signal.update(eyes_status == 'eye closed')
    yawn_signal.update(yawn_status == 'yawning')
    mobile_signal.update(result[0] == 0 and yolo_result.xyxy[0].shape[0] > 0)

    # Draw results on the original frame
    if not save:  # Only draw if we're displaying the frame
        cv2.putText(image, f'Driver eyes: {eyes_status}', (10,20), 0, 0.6,
                    conf.LM_COLOR, 1, lineType=cv2.LINE_AA)
        cv2.putText(image, f'Driver mouth: {yawn_status}', (10,40), 0, 0.6,
                    conf.CT_COLOR, 1, lineType=cv2.LINE_AA)
        if result[0] == 0 and yolo_result.xyxy[0].shape[0] > 0:
            cv2.putText(image, 'PHONE DETECTED!', (10,60), 0, 0.6,
                        (0,0,255), 2, lineType=cv2.LINE_AA)
    
    return image

def infer(args):
    try:
        setup_gpio()
        print("Starting DMS monitoring...")
        
        checkpoint = args.checkpoint
        
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=checkpoint)
        interpreter.allocate_tensors()

        # Optimize YOLO model further
        yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5n')
        yolo_model.classes = [67]  # phone class
        yolo_model.conf = 0.2      # Lower confidence threshold
        yolo_model.iou = 0.4       # Lower IoU threshold
        yolo_model.max_det = 1     # Only detect one phone
        yolo_model.agnostic = True # Non-class specific NMS
        
        # Optional: Set smaller inference size
        yolo_model.imgsz = [160, 160]  # Reduced detection size

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
                # Further reduce resolution and optimize camera settings
                cap.set(3, 240)  # Even smaller width
                cap.set(4, 180)  # Even smaller height
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FPS, 10)  # Further reduced FPS
                
                # Additional camera optimizations
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # Auto exposure
            
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
                    # Skip more frames
                    for _ in range(4):  # Skip 4 frames
                        cap.grab()
                
                image = infer_one_frame(image, interpreter, yolo_model, facial_tracker)
                
                if save:
                    out.write(image)
                else:
                    # Resize display window to be smaller
                    display_img = cv2.resize(image, (320, 240))
                    cv2.imshow('DMS', display_img)
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