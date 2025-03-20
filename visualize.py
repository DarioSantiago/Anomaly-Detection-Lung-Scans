import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for remote systems
import matplotlib.pyplot as plt
import os

def visualize_predictions(data_dir="../data", output_dir="../output", num_samples=5):
    """
    Loads and visualizes random test images alongside their predicted segmentation masks.
    """
    # Load test images and predictions
    test_images = np.load(os.path.join(data_dir, "scan_spring_2025.npy"))
    predictions = np.load(os.path.join(output_dir, "predictions.npy"))
    
    # Ensure shapes match
    assert test_images.shape[0] == predictions.shape[0], "Mismatch in number of images and predictions"
    
    # Debugging: Print unique classes in predictions
    unique_classes = np.unique(predictions)
    print(f"Unique classes in predicted masks: {unique_classes}\n")

    # Select random indices for visualization
    random_indices = np.random.choice(test_images.shape[0], num_samples, replace = False)
    
    plt.figure(figsize=(10, num_samples * 3))
    for i, idx in enumerate(random_indices):
        image = test_images[idx].squeeze()  # Remove extra dimensions
        mask = predictions[idx]  # Already in categorical format
        
        # Plot original image
        plt.subplot(num_samples, 2, 2 * i + 1)
        plt.imshow(image, cmap = 'gray')
        plt.axis('off')
        plt.title(f"Test Image {idx}")
        
        # Plot predicted segmentation mask
        plt.subplot(num_samples, 2, 2 * i + 2)
        plt.imshow(mask, cmap = 'jet', alpha = 0.6, vmin = 0, vmax = 3)  # Use 'jet' colormap for segmentation
        plt.axis('off')
        plt.title(f"Predicted Mask {idx}")
    
    plt.tight_layout()
    plt.savefig("../output/segmentation_results.png")
    print("Visualization saved as segmentation_results.png\n")


if __name__ == "__main__":
    visualize_predictions(num_samples = 5)
