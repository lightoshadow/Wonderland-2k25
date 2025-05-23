#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Wonderland Character Classifier - Webcam Inference with Frame Skipping
This script loads a trained EfficientNet model and performs real-time
inference using a webcam feed with frame skipping optimization.
"""

import cv2
import torch
import torchvision
import numpy as np
from PIL import Image
import time

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
    
    return model, weights

# Function to make predictions on an image
def predict_image(model, image, transform, device, class_names):
    # Convert OpenCV BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    
    # Apply transformations
    img_tensor = transform(pil_image).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.inference_mode():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    return class_names[predicted_class], confidence

def main():
    # Configuration
    model_path = "nuovi/WonderlandEfficentNet.pth"
    labels_file = "nuovi/labels.txt"
    
    # Frame skipping configuration
    FRAME_SKIP = 3  # Process every 3rd frame (skip 2 frames)
    CONFIDENCE_THRESHOLD = 0.5  # Only show predictions above this confidence
    PREDICTION_SMOOTHING = 5  # Number of frames to smooth predictions over
    
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
    cap = cv2.VideoCapture(0)  # 0 is usually the default webcam
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    print(f"Webcam initialized. Frame skipping: {FRAME_SKIP}. Press 'q' to quit.")
    
    # Variables for frame processing
    frame_count = 0
    current_prediction = "Initializing..."
    current_confidence = 0.0
    last_prediction_time = time.time()
    
    # Variables for prediction smoothing
    recent_predictions = []
    
    # Variables for FPS calculation
    prev_frame_time = 0
    inference_times = []
    
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture image")
            break
        
        # Calculate display FPS (actual frame rate)
        curr_frame_time = time.time()
        display_fps = 1.0 / (curr_frame_time - prev_frame_time) if prev_frame_time > 0 else 0
        prev_frame_time = curr_frame_time
        
        # Frame skipping logic - only process inference on certain frames
        should_process = (frame_count % FRAME_SKIP == 0)
        
        if should_process:
            # Record inference start time
            inference_start = time.time()
            
            # Make prediction
            try:
                predicted_class, confidence = predict_image(model, frame, transform, device, class_names)
                
                # Record inference time
                inference_time = time.time() - inference_start
                inference_times.append(inference_time)
                
                # Keep only last 30 inference times for averaging
                if len(inference_times) > 30:
                    inference_times.pop(0)
                
                # Update prediction if confidence is above threshold
                if confidence >= CONFIDENCE_THRESHOLD:
                    # Add to recent predictions for smoothing
                    recent_predictions.append((predicted_class, confidence))
                    
                    # Keep only recent predictions for smoothing
                    if len(recent_predictions) > PREDICTION_SMOOTHING:
                        recent_predictions.pop(0)
                    
                    # Find most common prediction in recent history
                    if recent_predictions:
                        # Get the most frequent prediction
                        pred_counts = {}
                        for pred, conf in recent_predictions:
                            if pred not in pred_counts:
                                pred_counts[pred] = []
                            pred_counts[pred].append(conf)
                        
                        # Find prediction with highest average confidence
                        best_pred = max(pred_counts.keys(), 
                                      key=lambda x: sum(pred_counts[x]) / len(pred_counts[x]))
                        best_conf = sum(pred_counts[best_pred]) / len(pred_counts[best_pred])
                        
                        current_prediction = best_pred
                        current_confidence = best_conf
                        last_prediction_time = time.time()
                
            except Exception as e:
                print(f"Inference error: {e}")
        
        # Age out old predictions (show "No detection" if prediction is too old)
        time_since_prediction = time.time() - last_prediction_time
        if time_since_prediction > 2.0:  # 2 seconds timeout
            current_prediction = "No recent detection"
            current_confidence = 0.0
        
        # Draw prediction on frame
        prediction_text = f"{current_prediction.replace('-', '')}: {current_confidence:.2f}"
        
        # Color code based on confidence
        if current_confidence > 0.8:
            color = (0, 255, 0)  # Green for high confidence
        elif current_confidence > 0.5:
            color = (0, 255, 255)  # Yellow for medium confidence
        else:
            color = (0, 0, 255)  # Red for low confidence
        
        cv2.putText(frame, prediction_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Draw performance metrics
        avg_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0
        inference_fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
        
        metrics_text = [
            f"Display FPS: {display_fps:.1f}",
            f"Inference FPS: {inference_fps:.1f}",
            f"Frame Skip: 1/{FRAME_SKIP}",
            f"Processing: {'YES' if should_process else 'NO'}"
        ]
        
        # Draw metrics in top right
        y_offset = 30
        for i, metric in enumerate(metrics_text):
            text_size = cv2.getTextSize(metric, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            x_position = frame.shape[1] - text_size[0] - 10
            cv2.putText(frame, metric, (x_position, y_offset + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw frame counter (for debugging)
        cv2.putText(frame, f"Frame: {frame_count}", (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
        
        # Display the frame
        cv2.imshow('Wonderland Character Classifier', frame)
        
        # Increment frame counter
        frame_count += 1
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Toggle frame skip rate with 's' key
            FRAME_SKIP = 1 if FRAME_SKIP > 1 else 3
            print(f"Frame skip changed to: 1/{FRAME_SKIP}")
        elif key == ord('t'):
            # Toggle confidence threshold with 't' key
            CONFIDENCE_THRESHOLD = 0.3 if CONFIDENCE_THRESHOLD > 0.3 else 0.7
            print(f"Confidence threshold changed to: {CONFIDENCE_THRESHOLD}")
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    
    # Print final statistics
    if inference_times:
        avg_inference_time = sum(inference_times) / len(inference_times)
        print(f"\nFinal Statistics:")
        print(f"Average inference time: {avg_inference_time:.3f}s")
        print(f"Average inference FPS: {1.0/avg_inference_time:.1f}")
        print(f"Total frames processed: {frame_count}")
        print(f"Inference frames: {frame_count // FRAME_SKIP}")

if __name__ == "__main__":
    main()
