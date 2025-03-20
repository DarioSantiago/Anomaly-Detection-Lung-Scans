"""
predict.py - Model Inference for Lung Segmentation

This script loads a trained U-Net model and performs segmentation on new lung scan images.
- Loads the trained model from `modeling_pipeline.pkl`.
- Loads new test images from `scan_spring_2025.npy`.
- Runs inference to predict segmentation masks.
- Saves the predicted masks to `predictions.npy`.

Acknowledgements:
This project is part of a Computer Science course on Data Mining and Medical Image Analysis.
Special thanks to Wake Forest University faculty for providing the dataset and guidance.

Author: Anthony Roca, Dario Santiago Lopez, and ChatGPT
Date: March 19, 2025
"""

import torch
import numpy as np
import os
from train import UNet  # Import the U-Net model
from torch.utils.data import TensorDataset, DataLoader

# ------------------ Load Model ------------------
def load_model(model_path="../output/modeling_pipeline.pkl"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location = device))
    model.eval()  # Set to evaluation mode
    print("Model loaded successfully!\n")
    return model, device

# ------------------ Run Inference ------------------
def predict(data_dir = "../data", output_file = "../output/predictions.npy", batch_size = 4):
    model, device = load_model()
    
    # Load test images
    test_images = np.load(os.path.join(data_dir, "scan_spring_2025.npy"))  # (100, 512, 512, 1)
    test_images = torch.tensor(test_images, dtype = torch.float32).permute(0, 3, 1, 2).to(device)  # (100, 1, 512, 512)
    
    # Convert test_images into a TensorDataset
    test_dataset = TensorDataset(test_images)
    test_loader = DataLoader(test_dataset, batch_size = batch_size)
    
    predictions = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch[0].to(device)
            output = model(batch)  # Forward pass
            pred_masks = torch.argmax(output, dim = 1).cpu().numpy()  # Convert to numpy
            predictions.extend(pred_masks)
    
    predictions = np.array(predictions)  # Shape: (100, 512, 512)
    np.save(output_file, predictions)
    print(f"Predictions saved to {output_file}\n")

    # Debugging
    print(f"Predictions shape: {predictions.shape}")
    unique_classes = np.unique(predictions)
    print(f"Unique classes in predictions: {unique_classes}\n")

if __name__ == "__main__":
    predict(batch_size = 4) # Adjust batch size as needed 
