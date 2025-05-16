#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Wonderland Character Classifier - Webcam Inference
This script loads a trained EfficientNet model and performs real-time
inference using a webcam feed.
"""

import cv2
import torch
import torchvision
import numpy as np
from PIL import Image

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
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    return class_names[predicted_class], confidence

def main():
    # Configuration
    model_path = "WonderlandEfficentNet.pth"
    labels_file = "labels.txt"
    
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
    
    print("Webcam initialized. Press 'q' to quit.")
    
    # Variables for FPS calculation
    prev_frame_time = 0
    curr_frame_time = 0
    
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture image")
            break
        
        # Calculate FPS
        curr_frame_time = cv2.getTickCount()
        fps = cv2.getTickFrequency() / (curr_frame_time - prev_frame_time) if prev_frame_time > 0 else 0
        prev_frame_time = curr_frame_time
        
        # Make prediction
        predicted_class, confidence = predict_image(model, frame, transform, device, class_names)
        
        # Draw prediction on frame
        prediction_text = f"{predicted_class.replace('-', '')}: {confidence:.2f}"
        cv2.putText(frame, prediction_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw FPS in top right
        fps_text = f"FPS: {fps:.1f}"
        fps_text_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        fps_x_position = frame.shape[1] - fps_text_size[0] - 10  # 10 pixels from right edge
        cv2.putText(frame, fps_text, (fps_x_position, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Display the frame
        cv2.imshow('Wonderland Character Classifier', frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

