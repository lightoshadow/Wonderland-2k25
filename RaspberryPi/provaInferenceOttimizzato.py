#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Wonderland Character Classifier - Webcam Inference
This script loads a trained EfficientNet model and performs real-time
inference using a webcam feed with performance optimizations.
"""

import cv2
import torch
import torchvision
import numpy as np
from PIL import Image
import time
from collections import deque

# Set device for inference
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load labels
def load_labels(labels_file):
    with open(labels_file, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    return labels

# Load the model
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
    
    # Optimize model for inference
    if device == "cuda":
        model = model.half()  # Use half precision (FP16) for CUDA
    
    return model, weights

class WebcamInference:
    def __init__(self, model, transform, class_names, device, 
                 input_size=(224, 224), skip_frames=1,
                 confidence_threshold=0.5, moving_avg_size=10):
        self.model = model
        self.transform = transform
        self.class_names = class_names
        self.device = device
        self.input_size = input_size
        self.skip_frames = skip_frames
        self.confidence_threshold = confidence_threshold
        
        # For performance tracking
        self.fps_queue = deque(maxlen=moving_avg_size)
        self.process_times = deque(maxlen=moving_avg_size)
        
        # For prediction smoothing
        self.pred_history = deque(maxlen=5)
        self.current_pred = None
        self.current_conf = 0.0
        
        # Frame counter
        self.frame_count = 0
    
    def preprocess_frame(self, frame):
        """Preprocess frame for inference with resize optimization"""
        # Resize frame to input size (much faster than letting transforms do it)
        resized = cv2.resize(frame, self.input_size)
        # Convert to RGB (PIL format)
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_frame)
        # Apply remaining transforms
        return self.transform(pil_image).unsqueeze(0).to(self.device)
    
    def predict(self, frame):
        """Make prediction on a frame"""
        self.frame_count += 1
        
        # Skip frames to improve performance
        if (self.frame_count - 1) % self.skip_frames != 0:
            return self.current_pred, self.current_conf, 0  # Return current prediction with no processing time
        
        start_time = time.time()
        
        # Preprocess frame efficiently
        img_tensor = self.preprocess_frame(frame)
        
        # Make prediction
        with torch.no_grad():  # Disable gradient calculation
            if device == "cuda":
                img_tensor = img_tensor.half()  # Use half precision if on CUDA
            
            outputs = self.model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            
            predicted_idx = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_idx].item()
        
        # Update prediction history for smoothing
        predicted_class = self.class_names[predicted_idx]
        self.pred_history.append((predicted_class, confidence))
        
        # Get most common prediction from history (simple smoothing)
        if len(self.pred_history) >= 3:  # Only start smoothing after collecting some predictions
            pred_counts = {}
            for pred, conf in self.pred_history:
                if pred not in pred_counts:
                    pred_counts[pred] = {"count": 0, "total_conf": 0}
                pred_counts[pred]["count"] += 1
                pred_counts[pred]["total_conf"] += conf
            
            # Find most common prediction with highest average confidence
            best_count = 0
            best_pred = None
            best_conf = 0
            
            for pred, data in pred_counts.items():
                if data["count"] > best_count:
                    best_count = data["count"]
                    best_pred = pred
                    best_conf = data["total_conf"] / data["count"]
                elif data["count"] == best_count and data["total_conf"] / data["count"] > best_conf:
                    best_pred = pred
                    best_conf = data["total_conf"] / data["count"]
            
            if best_pred:
                predicted_class = best_pred
                confidence = best_conf
        
        # Update current prediction
        self.current_pred = predicted_class
        self.current_conf = confidence
        
        # Calculate processing time
        process_time = time.time() - start_time
        self.process_times.append(process_time)
        
        return predicted_class, confidence, process_time
    
    def update_fps(self, frame_time):
        """Update FPS calculation"""
        self.fps_queue.append(1.0 / frame_time if frame_time > 0 else 0)
        return sum(self.fps_queue) / len(self.fps_queue) if self.fps_queue else 0
    
    def get_avg_process_time(self):
        """Get average processing time in milliseconds"""
        return sum(self.process_times) * 1000 / len(self.process_times) if self.process_times else 0
    
    def draw_stats(self, frame, fps, predicted_class, confidence):
        """Draw stats on frame"""
        h, w = frame.shape[:2]
        
        # Draw prediction on frame if confidence is above threshold
        if confidence >= self.confidence_threshold:
            prediction_text = f"{predicted_class.replace('-', '')}: {confidence:.2f}"
            cv2.putText(frame, prediction_text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw FPS in top right
        fps_text = f"FPS: {fps:.1f}"
        fps_text_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        fps_x_position = w - fps_text_size[0] - 10  # 10 pixels from right edge
        cv2.putText(frame, fps_text, (fps_x_position, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Draw processing time in ms
        if self.process_times:
            process_text = f"Process: {self.get_avg_process_time():.1f}ms"
            cv2.putText(frame, process_text, (fps_x_position, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return frame

def main():
    # Configuration parameters
    model_path = "WonderlandEfficentNet.pth"
    labels_file = "labels.txt"
    webcam_index = 0
    frame_width = 640   # Reduced frame size for better performance
    frame_height = 480
    skip_frames = 2     # Process every N frames
    confidence_threshold = 0.5  # Minimum confidence to show prediction
    
    # Load class names
    class_names = load_labels(labels_file)
    print(f"Loaded {len(class_names)} classes: {class_names}")
    
    # Load model and get weights
    model, weights = load_model(model_path, len(class_names))
    
    # Get transforms from the weights
    transform = weights.transforms()
    print("Using transforms from model weights")
    
    # Initialize webcam
    print("Initializing webcam...")
    cap = cv2.VideoCapture(webcam_index)
    
    # Set lower resolution for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    print("Webcam initialized. Press 'q' to quit.")
    
    # Create inference handler
    inference = WebcamInference(
        model=model,
        transform=transform,
        class_names=class_names,
        device=device,
        skip_frames=skip_frames,
        confidence_threshold=confidence_threshold
    )
    
    prev_frame_time = time.time()
    avg_fps = 0
    
    try:
        while True:
            # Read frame from webcam
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Failed to capture image")
                break
            
            # Calculate real FPS (capture rate)
            curr_frame_time = time.time()
            frame_time = curr_frame_time - prev_frame_time
            prev_frame_time = curr_frame_time
            
            # Update average FPS
            avg_fps = inference.update_fps(frame_time)
            
            # Make prediction
            predicted_class, confidence, _ = inference.predict(frame)
            
            # Draw stats on frame
            display_frame = inference.draw_stats(frame, avg_fps, predicted_class, confidence)
            
            # Display the frame
            cv2.imshow('Wonderland Character Classifier', display_frame)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("Interrupted by user")
    
    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        print("Application closed")

if __name__ == "__main__":
    main()
