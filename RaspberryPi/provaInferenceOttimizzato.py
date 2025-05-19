#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Optimized Wonderland Character Classifier - Webcam Inference
This script loads a trained EfficientNet model and performs real-time
inference using a webcam feed with various optimizations.
"""

import cv2
import torch
import torchvision
import numpy as np
import threading
import queue
import time
from PIL import Image

# Set device for inference
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Enable mixed precision if using CUDA
use_amp = device == "cuda"
if use_amp:
    print("Enabling automatic mixed precision")

# Configuration
MODEL_PATH = "WonderlandEfficentNet.pth"
LABELS_FILE = "labels.txt"
FRAME_SKIP = 1  # Process every nth frame
INFERENCE_SIZE = (224, 224)  # Standard size for EfficientNet
DISPLAY_FPS = True
MAX_QUEUE_SIZE = 3

# Load labels
def load_labels(labels_file):
    with open(labels_file, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    return labels

# Load and optimize the model
def load_model(model_path, num_classes):
    # Create the EfficientNet model with the default weights
    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    model = torchvision.models.efficientnet_b0(weights=weights)
    
    # Modify the classifier for our number of classes
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True),
        torch.nn.Linear(in_features=1280, out_features=num_classes, bias=True)
    )
    
    # Load the trained state dict
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Optimize with TorchScript (JIT compilation)
    try:
        example_input = torch.rand(1, 3, 224, 224).to(device)
        model = torch.jit.trace(model, example_input)
        print("Model optimized with TorchScript")
    except Exception as e:
        print(f"JIT compilation failed: {e}")
    
    # Attempt model quantization for CPU
    if device == "cpu":
        try:
            model_quantized = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
            model = model_quantized
            print("Model quantized for CPU inference")
        except Exception as e:
            print(f"Quantization failed: {e}")
    
    return model, weights

# Efficient preprocessing
def preprocess_image(image, transform):
    """Efficiently preprocess an image for inference"""
    # Resize image directly with OpenCV (more efficient than PIL)
    resized = cv2.resize(image, INFERENCE_SIZE)
    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    # Convert to tensor and normalize in one step
    img_tensor = transform(Image.fromarray(rgb_image)).unsqueeze(0)
    return img_tensor

# Prediction function with mixed precision support
def predict_image(model, img_tensor, class_names):
    img_tensor = img_tensor.to(device)
    
    # Use mixed precision for CUDA
    if use_amp:
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                outputs = model(img_tensor)
    else:
        with torch.no_grad():
            outputs = model(img_tensor)
    
    # Get prediction
    probabilities = torch.softmax(outputs, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    confidence = probabilities[0][predicted_class].item()
    
    return class_names[predicted_class], confidence

# Frame capture worker
def capture_frames(cap, frame_queue, stop_event):
    frame_count = 0
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            stop_event.set()
            break
            
        # Skip frames if needed
        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue
            
        # If queue is full, remove oldest item
        if frame_queue.qsize() >= MAX_QUEUE_SIZE:
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
                
        frame_queue.put((frame, time.time()))
        
        # Small sleep to prevent CPU hogging
        time.sleep(0.001)

# Inference worker
def process_frames(model, transform, class_names, frame_queue, result_queue, stop_event):
    while not stop_event.is_set():
        try:
            frame, timestamp = frame_queue.get(timeout=1.0)
            
            # Preprocess
            img_tensor = preprocess_image(frame, transform)
            
            # Predict
            predicted_class, confidence = predict_image(model, img_tensor, class_names)
            
            # Add result to queue
            result_queue.put((frame, predicted_class, confidence, timestamp))
            
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error in inference: {e}")
            continue

def main():
    # Load class names
    class_names = load_labels(LABELS_FILE)
    print(f"Loaded {len(class_names)} classes: {class_names}")
    
    # Load model and get weights
    model, weights = load_model(MODEL_PATH, len(class_names))
    
    # Get transforms from the weights
    transform = weights.transforms()
    
    # Initialize webcam
    print("Initializing webcam...")
    cap = cv2.VideoCapture(0)
    
    # Try to set higher resolution if available
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    print("Webcam initialized. Press 'q' to quit.")
    
    # Create queues for frame passing
    frame_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
    result_queue = queue.Queue()
    
    # Create stop event for signaling threads to exit
    stop_event = threading.Event()
    
    # Start worker threads
    capture_thread = threading.Thread(target=capture_frames, 
                                     args=(cap, frame_queue, stop_event))
    inference_thread = threading.Thread(target=process_frames,
                                       args=(model, transform, class_names, 
                                             frame_queue, result_queue, stop_event))
    
    capture_thread.daemon = True
    inference_thread.daemon = True
    
    capture_thread.start()
    inference_thread.start()
    
    # Variables for FPS calculation
    frame_times = []
    fps_update_interval = 0.5  # seconds
    last_fps_update = time.time()
    current_fps = 0
    
    # Display loop
    last_result = None
    try:
        while True:
            # Get the latest result if available
            try:
                frame, predicted_class, confidence, timestamp = result_queue.get_nowait()
                last_result = (frame, predicted_class, confidence, timestamp)
                
                # Update FPS
                current_time = time.time()
                frame_times.append(current_time)
                
                # Remove old frame times
                while frame_times and frame_times[0] < current_time - 1.0:
                    frame_times.pop(0)
                
                # Update FPS counter periodically
                if current_time - last_fps_update > fps_update_interval:
                    current_fps = len(frame_times)
                    last_fps_update = current_time
                
            except queue.Empty:
                # If no new result, use the last one
                if last_result is None:
                    # If there's no result yet, wait a bit
                    time.sleep(0.01)
                    continue
                frame, predicted_class, confidence, timestamp = last_result
            
            # Create a copy of the frame for drawing
            display_frame = frame.copy()
            
            # Draw prediction on frame
            prediction_text = f"{predicted_class.replace('-', '')}: {confidence:.2f}"
            cv2.putText(display_frame, prediction_text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Draw FPS in top right
            if DISPLAY_FPS:
                fps_text = f"FPS: {current_fps}"
                fps_text_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                fps_x_position = display_frame.shape[1] - fps_text_size[0] - 10
                cv2.putText(display_frame, fps_text, (fps_x_position, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Display the frame
            cv2.imshow('Wonderland Character Classifier', display_frame)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        # Clean up
        print("Cleaning up resources...")
        stop_event.set()
        
        # Wait for threads to finish
        capture_thread.join(timeout=1.0)
        inference_thread.join(timeout=1.0)
        
        cap.release()
        cv2.destroyAllWindows()
        print("Application terminated.")

if __name__ == "__main__":
    main()
