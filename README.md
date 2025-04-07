# COVID-19 Detection from Chest X-Ray Images using Deep Learning

## Author Information  
**Full Name:** Chandrika Patibandla  
**University Code:** 700777118  

## Overview  
This project aims to build and evaluate deep learning models to classify chest X-ray images for COVID-19 detection. The goal is to compare baseline architectures and explore improvements through custom CNN designs for enhanced diagnostic performance, assisting healthcare professionals in early diagnosis and reducing the transmission of the virus.

### Tasks

- **Task 1:**  
  - Re-implement and evaluate the following baseline models:  
    - A basic Convolutional Neural Network (CNN)  
    - MobileNet  
    - VGG16  

- **Task 2:**  
  - Develop a custom CNN model with deeper architecture.  
  - Add more convolutional layers along with max pooling and dropout layers to improve model performance.  

## Dataset  
The dataset for this project is available at [COVID-19 Chest X-Ray Dataset](https://www.kaggle.com/datasets/fusicfenta/chest-xray-for-covid19-detection). It contains three main folders:
1. **Training Folder**: Contains chest X-ray images for training the model. The images are divided into two subfolders: **Normal** (X-rays of healthy patients) and **COVID** (X-rays of COVID-19 positive patients).
2. **Validation Folder**: Contains images for validating the model’s performance. This folder also has the **Normal** and **COVID** subfolders.
3. **Testing Folder**: Contains images for testing the model once it has been trained.

The images in these folders are resized to 224x224 pixels for input into the CNN model.

## Requirements  
In order to run this code, the following Python libraries are required:
- **Keras**: Used to create and train the CNN model.
- **Matplotlib**: For visualizing training progress, loss, and accuracy graphs.
- **Pandas**: For loading and handling dataset information.
- **TensorFlow**: Acts as the backend for Keras, handling the computation of the model.
- **Scipy**: For scientific and numerical computations.
- **Scikit-learn**: For utilities related to machine learning, such as splitting datasets and evaluating model performance.
- **Seaborn**: For statistical data visualization, especially useful for plots like heatmaps.
- **OpenCV**: For image loading and displaying predictions on images.

## Model Flow  
The model follows these steps:

1. **Loading Dataset with Augmentation**:  
   The dataset is loaded using Keras' `ImageDataGenerator`. The training dataset includes data augmentation techniques like zoom, shear, and horizontal flip to increase model robustness, while the validation dataset only applies rescaling. 

2. **Model Architecture**:  
   The model is a Convolutional Neural Network (CNN) consisting of multiple convolutional layers to extract key features from the X-ray images. After convolution, the model uses max-pooling layers to reduce spatial dimensions, followed by dropout layers to prevent overfitting. Finally, the model includes fully connected layers for classification.

3. **Baseline Models (Task 1)**:  
   - Re-implementation of the CNN, MobileNet, and VGG16 architectures based on the reference paper.  
   - These models are compared to assess their effectiveness in COVID-19 detection from chest X-rays.

4. **Custom Model (Task 2)**:  
   A deeper CNN architecture is developed with additional layers to enhance performance:
   - More convolutional layers with max pooling and dropout to prevent overfitting.  
   - Aim to improve the accuracy and robustness of the model compared to the baseline models.

5. **Training the Model**:  
   The CNN model is trained using the training dataset, and it undergoes evaluation on the validation dataset. The training process is carried out for a fixed number of epochs (10 in this case) to allow the model to learn patterns from the data.

6. **Evaluation**:  
   Once the model is trained, it is evaluated using the test dataset. This step helps in assessing the model's performance based on various metrics like accuracy, precision, recall, and confusion matrix.

## Testing the Model  
After training, the model is ready to be tested on new chest X-ray images. Here’s how the testing process works:
- **Load the Model**: Load the trained model.
- **Use Test Images**: Use new test images (in the testing folder) to make predictions.
- **Make Predictions**: After preprocessing the test images (resizing to 224x224), pass them through the model to classify them as either **Normal** or **COVID**.
- **Display Results**: The model’s predictions can be displayed using OpenCV to visualize whether the image is classified as COVID-19 positive or normal.
- **Confusion Matrix**: A confusion matrix is generated to evaluate model performance by showing the true positives, false positives, true negatives, and false negatives.

## Source Code  
The code was developed in Google Colab. To view the results, please ensure that all cells are executed.
