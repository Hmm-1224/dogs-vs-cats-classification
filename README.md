# dogs-vs-cats-classification

This project demonstrates building and training a Convolutional Neural Network (CNN) to classify images of cats and dogs. The model leverages pre-trained MobileNetV2 for feature extraction, combined with custom convolutional and dense layers for the classification task.

**Project Structure**
1. Dataset Handling
Downloads and unzips the Cats vs. Dogs dataset.
Augments training data using ImageDataGenerator.
Normalizes validation data for better performance.
2. Model Architecture
Base Model: Uses MobileNetV2 as a frozen base for feature extraction.
Custom Layers:
Convolutional layers with Batch Normalization and Max Pooling.
Dense layers with Dropout for regularization.
Output layer with a sigmoid activation function for binary classification.
3. Training and Evaluation
Loss function: binary_crossentropy.
Optimizer: Adam.
Includes EarlyStopping to prevent overfitting by monitoring validation loss.
Generates plots of accuracy and loss to visualize training progress.
4. Testing
The trained model predicts whether a given image is a cat (label 0) or a dog (label 1).
Steps to Run the Code
Clone the Repository: Clone this repository or copy the script.
Authenticate Dataset Download:
Upload your Kaggle API token (kaggle.json) to authenticate and download the dataset.
Install Dependencies:
pip install tensorflow opencv-python matplotlib kaggle
Run the Script: Execute the script in a Jupyter Notebook or Google Colab environment.
Test the Model: Provide a test image of a cat or dog to validate predictions.
**Key Features**
Data Augmentation: Improves model generalization by applying transformations like rotation, zoom, and flips.
Batch Normalization: Speeds up training and provides regularization.
Dropout Layers: Reduces overfitting by randomly deactivating neurons during training.
Pre-trained MobileNetV2: Enhances performance by utilizing transfer learning.
Early Stopping: Saves time and prevents overfitting by stopping training when validation loss stops improving.

**Model Evaluation**<br>
*Observations:*<br>
Without L2 regularization, the model performs better and avoids underfitting.
Graphs of training and validation metrics help identify overfitting issues.<br>
*Visualization*<br>
Accuracy and Loss Curves: Displayed for both training and validation data to aid in analysis.
*Testing the Model*
Load a test image (cat or dog).
Resize it to 256x256 pixels.
Pass the image through the model for prediction.
Interpret the output:
0: Cat.
1: Dog.
Example Prediction
Input: Cat Image
Output: [[0.]]

Input: Dog Image
Output: [[1.]]

References
Dataset: Kaggle - Dogs vs. Cats
