#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Wonderland Character Classifier - Webcam Inference
This script loads a trained quantized EfficientNet model and performs real-time
inference using a webcam feed.
"""

import cv2
import torch
import torchvision
import numpy as np
from PIL import Image

# Set device for inference - quantized models typically run on CPU
device = "cpu"  # Force CPU for quantized model
print(f"Using device: {device}")

# Load labels
def load_labels(labels_file):
    with open(labels_file, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    return labels

# Load the quantized model
def load_quantized_model(model_path, num_classes):
    # First create the base model with the same architecture
    model = torchvision.models.efficientnet_b0(weights=None)
    
    # Modify the classifier for our number of classes
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True),
        torch.nn.Linear(in_features=1280, out_features=num_classes, bias=True)
    )
    
    # Set to eval mode before quantization
    model.eval()
    
    # Apply dynamic quantization to the model (same quantization used during saving)
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear, torch.nn.Conv2d},
        dtype=torch.qint8
    )
    
    # Load the quantized state dict
    quantized_model.load_state_dict(torch.load(model_path, map_location=device))
    quantized_model.eval()
    
    return quantized_model

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
    model_path = "WonderlandEfficientNet_quantized.pth"
    labels_file = "labels.txt"
    
    # Load class names
    class_names = load_labels(labels_file)
    print(f"Loaded {len(class_names)} classes: {class_names}")
    
    # Load quantized model
    model = load_quantized_model(model_path, len(class_names))
    print("Loaded quantized model")
    
    # Create transforms (can't use weights.transforms() with quantized model)
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(224),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    print("Created transforms for inference")
    
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