
"""
utils.py - Data Loading & Preprocessing for Lung Segmentation

This script handles loading and preprocessing of lung scan images and their corresponding labels.
- Loads images and labels from .npy files.
- Converts one-hot encoded labels to categorical format.
- Normalizes image pixel values.
- Saves the preprocessed data for training.

Acknowledgements:
This project is part of a Computer Science course on Data Mining and Medical Image Analysis.
Special thanks to Wake Forest University faculty for providing the dataset and guidance.

Authors: Anthony Roca, Dario Santiago Lopez, and ChatGPT

Date: March 19, 2025
"""
import numpy as np 
import os 

def load_data(data_dir = "../data"):
    """
    Loads and preprocesses lung scan images and labels.
    - Converts one-hot encoded labels to categorical format.
    - Normalizes image pixel values.
    """
    
    print("Loading data...\n")

    # Load imagaes and labels 
    scan_images = np.load(os.path.join(data_dir, "scan_fall_2019.npy")) # Shape: (num_images, height, width)
    scan_labels = np.load(os.path.join(data_dir, "labels_fall_2019.npy")) # Shape: (num_images, height, width)

    # Convert the one-hot encoded labels to categorical labels (0, 1, 2, 3)
    cat_labels = np.argmax(scan_labels, axis = -1)  

    # Normalize the images 
    scan_images = scan_images.astype(np.float32) / 255.0     # Scale to [0, 1]

    print(f"Loaded {scan_images.shape[0]} images with shape {scan_images.shape[1:]}.")
    print(f"Labels converted to categorical format with shape {cat_labels.shape}.")
    # print(f"Cateogorical labels shape: {cat_labels.shape}") # used for debugging 

    # Debugging 
    unique_labels, counts = np.unique(cat_labels, return_counts = True)
    class_distribution = dict(zip(unique_labels, counts))
    print(f"Class distribution in the training labels: {class_distribution}\n") 

    return scan_images, cat_labels 


def check_label_distribution(data_dir="../data"):
    """
    Checks the class distribution in training labels. Skips test labels if not available.
    """
    # Load training labels
    train_labels = np.load(os.path.join(data_dir, "labels_fall_2019.npy"))
    train_class_counts = np.unique(train_labels, return_counts=True)
    print(f"Training label distribution: {dict(zip(train_class_counts[0], train_class_counts[1]))}")

    #  Skip test label check if the file doesn't exist
    test_label_path = os.path.join(data_dir, "labels_spring_2025.npy")
    if os.path.exists(test_label_path):
        test_labels = np.load(test_label_path)
        test_class_counts = np.unique(test_labels, return_counts=True)
        print(f"Test label distribution: {dict(zip(test_class_counts[0], test_class_counts[1]))}")
    else:
        print("Test labels not found. Predictions will be manually evaluated.\n")

if __name__ == "__main__": 
    images, labels = load_data()
    np.save("../data/preprocessed_images.npy", images)
    np.save("../data/preprocessed_labels.npy", labels)
    print("Preprocessed data saved successfully!\n")

    # Run label distribution check
    check_label_distribution()

