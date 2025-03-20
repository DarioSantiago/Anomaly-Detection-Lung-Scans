"""
train.py - U-Net Model Training for Lung Segmentation

This script trains a U-Net model to segment suspicious regions in lung scan images.
- Loads preprocessed images and labels.
- Defines the U-Net architecture for image segmentation.
- Uses CrossEntropyLoss for multi-class classification.
- Saves the trained model for later inference.

Acknowledgements:
This project is part of a Computer Science course on Data Mining and Medical Image Analysis.
Special thanks to Wake Forest University faculty for providing the dataset and guidance.

Author: Anthony Roca, Dario Santiago Lopez, and ChatGPT 
Date: March 19, 2025
"""

import torch 
import torch.nn as nn 
import torch.optim as optim 
import numpy as np 
import os 
from torch.utils.data import DataLoader, Dataset 
import torch.nn.functional as F 
import torchvision.transforms as transforms 

# ------------------ Data Augmentation ------------------
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # Randomly flip images
    transforms.RandomRotation(10),  # Rotate by ±10 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Adjust brightness/contrast
])

# ------------------ U-Net Model ------------------
class UNet(nn.Module): 
    def __init__(self, in_channels = 1, out_channels = 4): 
        super(UNet, self).__init__() 

        def conv_block(in_c, out_c): 
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size = 3, padding = 1), 
                nn.ReLU(), 
                nn.Conv2d(out_c, out_c, kernel_size = 3, padding = 1), 
                nn.ReLU()
            )
        
        # Encoder(s) 
        self.encoder1 = conv_block(in_channels, 64) 
        self.encoder2 = conv_block(64, 128) 
        self.encoder3 = conv_block(128, 256) 
        self.encoder4 = conv_block(256, 512)   
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)      # Kernel size = 2, stride = 2
        # Bottleneck
        self.bottleneck = conv_block(512, 1024) 
        # Decoder(s)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = conv_block(1024, 512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = conv_block(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = conv_block(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = conv_block(128, 64)
        
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x): 
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))
        
        b = self.bottleneck(self.pool(e4))
        
        d4 = self.decoder4(torch.cat((self.upconv4(b), e4), dim = 1))
        d3 = self.decoder3(torch.cat((self.upconv3(d4), e3), dim = 1))
        d2 = self.decoder2(torch.cat((self.upconv2(d3), e2), dim = 1))
        d1 = self.decoder1(torch.cat((self.upconv1(d2), e1), dim = 1))
        
        return self.final_conv(d1)


# ------------------ Custom Dataset ------------------
class LungSegmentationDataset(Dataset): 
    def __init__(self, image_file, label_file, subset = 1.0, augment = False): # Subset parameter is to take a subset of the data for training (for time efficiency) 
        self.images = np.load(image_file)
        self.labels = np.load(label_file) 
        self.augment = augment

        # Convert to PyTorch tensors 
        self.images = torch.tensor(self.images, dtype = torch.float32).permute(0, 3, 1, 2)   # Change to (num_images, channels, height, width)
        self.labels = torch.tensor(self.labels, dtype = torch.long)                          # Change to (num_images, height, width)

        # Reduce the dataset size for training (soley for time efficiency) 
        num_samples = int(len(self.images) * subset)
        indices = torch.randperm(len(self.images))[:num_samples] # Randomly select the subset 
        self.images = self.images[indices]
        self.labels = self.labels[indices]
        print(f"Using {num_samples} samples for training.\n")


    def __len__(self): 
        return len(self.images) 

    def __getitem__(self, idx): 
        image, label = self.images[idx], self.labels[idx]
        if self.augment:
            image = train_transforms(image)
        return image, label

# ------------------ Training Pipeline ------------------ 
def train_model(data_dir = "../data", epochs = 10, batch_size = 4, lr = 0.001, subset = 1.0): 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    dataset = LungSegmentationDataset( 
        os.path.join(data_dir, "preprocessed_images.npy"), 
        os.path.join(data_dir, "preprocessed_labels.npy"), 
        subset=subset, 
        augment=True
    )
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True)

    model = UNet().to(device) 
    
    # Compute class weights for CrossEntropyLoss since the dataset is imbalanced 
    labels = np.load(os.path.join(data_dir, "preprocessed_labels.npy"))
    unique_labels, counts = np.unique(labels, return_counts = True)
    total_pixels = sum(counts) 

    # Compute inverse frequency for each class 
    class_weights = torch.tensor([total_pixels / (count + 1e-6) for count in counts], dtype = torch.float32).to(device)
    print(f"Using class weights: {class_weights.cpu().numpy()}\n")

    # Apply weights to CrossEntropyLoss
    criterion = nn.CrossEntropyLoss(weight = class_weights)
    optimizer = optim.Adam(model.parameters(), lr = lr)

    print("Starting training...")

    for epoch in range(epochs): 
        model.train()
        epoch_loss = 0.0 
        for images, batch_labels in dataloader: 
            images, batch_labels = images.to(device), batch_labels.to(device) 
            optimizer.zero_grad() 
            outputs = model(images) 
            loss = criterion(outputs, batch_labels) 
            loss.backward() 
            optimizer.step() 
            epoch_loss += loss.item() 
        
        print(f"Epoch [{epoch + 1} / {epochs}], Loss: {epoch_loss / len(dataloader):.4f}")

        # Debugging:Check training predictions 
        check_predictions(model, dataloader, device)
    
    # Save the trained model 
    torch.save(model.state_dict(), "../output/modeling_pipeline.pkl")
    print("Model saved successfully!")

# ------------------ Debugging Function(s) ------------------
def check_predictions(model, dataloader, device): 
    model.eval() 
    try:
        sample_images, _ = next(iter(dataloader))  # Get a small batch of images  
    except StopIteration:
        print("Dataloader is empty, skipping prediction check.")
        return
    
    sample_images = sample_images.to(device)

    with torch.no_grad(): 
        output = model(sample_images)
        pred_classes = torch.argmax(output, dim = 1).cpu().numpy() # Get the predicted classes 
        unique_classes, counts = np.unique(pred_classes, return_counts = True)
    
    print(f"Debugging Model Predictions:")
    print(f"- Prediction shape: {pred_classes.shape}")  # Expected (batch_size, height, width)
    print(f"- Unique classes in batch predictions: {unique_classes}")
    print(f"- Class counts: {dict(zip(unique_classes, counts))}\n")

if __name__ == "__main__": 
    train_model(epochs = 10, batch_size = 4, lr = 0.001, subset = 1.0)
    print("Training complete!\n")
