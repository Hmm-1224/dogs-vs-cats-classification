# dogs-vs-cats-classification

This project demonstrates building and training a Convolutional Neural Network (CNN) to classify images of cats and dogs. The model leverages pre-trained MobileNetV2 for feature extraction, combined with custom convolutional and dense layers for the classification task.

**Project Structure**
1. Dataset Handling<br>
Downloads and unzips the Cats vs. Dogs dataset.
Augments training data using ImageDataGenerator.
Normalizes validation data for better performance.
2. Model Architecture<br>
Base Model: Uses MobileNetV2 as a frozen base for feature extraction.
Custom Layers:
Convolutional layers with Batch Normalization and Max Pooling.
Dense layers with Dropout for regularization.
Output layer with a sigmoid activation function for binary classification.
3. Training and Evaluation<br>
Loss function: binary_crossentropy.<br>
Optimizer: Adam.<br>
Includes EarlyStopping to prevent overfitting by monitoring validation loss.<br>
Generates plots of accuracy and loss to visualize training progress.<br>
4. Testing<br>
The trained model predicts whether a given image is a cat (label 0) or a dog (label 1).
*Steps to Run the Code:<br>
Clone the Repository: Clone this repository or copy the script.<br>
*Authenticate Dataset Download:<br>
Upload your Kaggle API token (kaggle.json) to authenticate and download the dataset.<br>
*Install Dependencies:<br>
pip install tensorflow opencv-python matplotlib kaggle<br>
*Run the Script: Execute the script in a Jupyter Notebook or Google Colab environment.<br>
*Test the Model: Provide a test image of a cat or dog to validate predictions.<br>
**Key Features**<br>
-Data Augmentation: Improves model generalization by applying transformations like rotation, zoom, and flips.<br>
-Batch Normalization: Speeds up training and provides regularization.<br>
-Dropout Layers: Reduces overfitting by randomly deactivating neurons during training.<br>
-Pre-trained MobileNetV2: Enhances performance by utilizing transfer learning.<br>
-Early Stopping: Saves time and prevents overfitting by stopping training when validation loss stops improving.<br>

**Model Evaluation**<br>
*Observations:*<br>
Without L2 regularization, the model performs better and avoids underfitting.
Graphs of training and validation metrics help identify overfitting issues.<br>
*Visualization:*<br>
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
