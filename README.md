# dogs-vs-cats-classification
Cats vs. Dogs Classification with TensorFlow
This project demonstrates building and training a Convolutional Neural Network (CNN) to classify images of cats and dogs. The model leverages pre-trained MobileNetV2 for feature extraction, combined with custom convolutional and dense layers for the classification task.

Project Structure

Dataset Handling:
Downloads and unzips the Cats vs. Dogs dataset.
Augments training data using ImageDataGenerator.
Normalizes validation data for better performance.

Model Architecture:
Uses MobileNetV2 as a frozen base for feature extraction.
Adds custom layers:
Convolutional layers with Batch Normalization and Max Pooling.
Dense layers with Dropout for regularization.
Output layer uses a sigmoid activation function for binary classification.

Training and Evaluation:
Trains using binary_crossentropy loss and the Adam optimizer.
Includes EarlyStopping to prevent overfitting by monitoring validation loss.
Generates plots of accuracy and loss to visualize training progress.
Testing:

The trained model predicts whether a given image is a cat (label 0) or a dog (label 1).
Steps to Run the Code:
Clone this repository or copy the script.
Upload your Kaggle API token (kaggle.json) to authenticate and download the dataset.
Install dependencies:
bash
Copy code
pip install tensorflow opencv-python matplotlib kaggle
Run the script in a Jupyter Notebook or Google Colab environment.
Provide a test image of a cat or dog to validate predictions.

Key Features
Data Augmentation: Improves model generalization by applying transformations like rotation, zoom, and flips.
Batch Normalization: Speeds up training and provides regularization.
Dropout Layers: Reduces overfitting by randomly deactivating neurons during training.
Pre-trained MobileNetV2: Enhances performance by utilizing transfer learning.
Early Stopping: Saves time and prevents overfitting by stopping training when validation loss stops improving.

Model Evaluation
Observations:
Without L2 regularization, the model performs better and avoids underfitting.
Graphs of training and validation metrics help identify overfitting issues.
Visualization:
Accuracy and loss curves for both training and validation data are displayed for analysis.
Testing the Model
Load a test image (cat or dog).
Resize it to 256x256 pixels.
Pass the image through the model for prediction.
Interpret the output:
0: Cat
1: Dog
Example Prediction
Input: Cat Image
Output: [[0.]]

Input: Dog Image
Output: [[1.]]

References
Dataset: Kaggle - Dogs vs. Cats
Pre-trained Model: MobileNetV2
