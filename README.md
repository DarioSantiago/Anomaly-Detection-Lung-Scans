🫁 Lung CT Scan Segmentation (U-Net Model)
📌 Project Overview

This project implements a U-Net deep learning architecture to perform semantic segmentation on lung CT scan images. The goal is to classify each pixel into specific lung regions to aid in medical image analysis, particularly in detecting areas of interest for further examination.

The project was completed as part of a Computer Science Data Mining & Medical Image Analysis course at Wake Forest University.

🚀 Features

Custom U-Net Model: Fully implemented in PyTorch for multi-class segmentation.

Pixel-Wise Accuracy Tracking: Evaluates segmentation performance each epoch.

Hyperparameter Tuning: Experiments with batch sizes, learning rates, and epochs.

Data Augmentation Support: Improves generalization with random flips, rotations, and brightness/contrast adjustments.

Segmentation Visualization: Generates color-coded prediction masks for qualitative evaluation.

Class Distribution Analysis: Tracks predicted class percentages to detect imbalances.

📊 Final Model Performance

Best Configuration:

Batch Size: 4

Learning Rate: 0.0005

Epochs: 10

Batch Size	Learning Rate	Epochs	Final Loss	Final Accuracy
4	0.001	10	0.4139	84.37%
8	0.001	10	0.4496	79.74%
4	0.0005	10	0.2848	93.05%

Final Prediction Class Distribution:

Class 0 (Background): 0.23%

Class 1: 0% (likely due to low representation in training data and possible class weighting issues)

Class 2 (Lung Tissue): 0.93%

Class 3 (Lung ROI): 98.84%

🖼 Sample Segmentation Output

📂 Project Structure
.
├── scripts/
│   ├── assignment_4.py       # Main training script
│   ├── predict.py            # Model inference script
│   ├── visualize.py          # Visualization of predictions
│   ├── utils.py              # Data loading and preprocessing
├── data/
│   ├── preprocessed_images.npy
│   ├── preprocessed_labels.npy
├── output/
│   ├── predictions.npy
│   ├── segmentation_results.png
│   ├── modeling_pipeline.pkl
├── Assignment 4 Report.pdf   # Full project report

🛠 How to Run
1️⃣ Preprocess Data
python scripts/utils.py

2️⃣ Train Model
python scripts/assignment_4.py

3️⃣ Generate Predictions
python scripts/predict.py

4️⃣ Visualize Predictions
python scripts/visualize.py

📌 Future Improvements

Balance the dataset or apply more aggressive class weighting to recover missing class predictions.

Implement Dice loss or IoU loss for better segmentation boundary accuracy.

Explore transfer learning with pretrained encoder backbones.

Authors: Anthony Roca, Dario Santiago Lopez, and ChatGPT
Date: March 19, 2025
