import numpy as np 
import os 

def load_data(data_dir = "../data"):

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

    return scan_images, cat_labels 

if __name__ == "__main__": 
    images, labels = load_data()
    np.save("../data/preprocessed_images.npy", images)
    np.save("../data/preprocessed_labels.npy", labels)
    print("Preprocessed data saved successfully!\n")

