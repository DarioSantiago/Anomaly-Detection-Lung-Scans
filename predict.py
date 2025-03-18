import torch
import numpy as np
import os
from train import UNet  # Import the U-Net model

# ------------------ Load Model ------------------
def load_model(model_path="../output/modeling_pipeline.pkl"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set to evaluation mode
    print("Model loaded successfully!\n")
    return model, device

# ------------------ Run Inference ------------------
def predict(data_dir="../data", output_file="../output/predictions.npy"):
    model, device = load_model()
    
    # Load test images
    test_images = np.load(os.path.join(data_dir, "scan_spring_2025.npy"))  # (100, 512, 512, 1)
    test_images = torch.tensor(test_images, dtype=torch.float32).permute(0, 3, 1, 2).to(device)  # (100, 1, 512, 512)
    
    predictions = []
    with torch.no_grad():
        for img in test_images:
            img = img.unsqueeze(0)  # Add batch dimension
            output = model(img)  # Forward pass
            pred_mask = torch.argmax(output, dim = 1).squeeze(0).cpu().numpy()  # Convert to numpy
            predictions.append(pred_mask)
    
    predictions = np.array(predictions)  # Shape: (100, 512, 512)
    np.save(output_file, predictions)
    print(f"Predictions saved to {output_file}\n")

if __name__ == "__main__":
    predict()