# AI Face Recognition System

This project is a real time AI face recognition system developed using Python and several machine learning libaries. The system processes images to detect and recognize faces using deep learning and neural network techniques. Below are the steps we have follwoed and the libraries used in the development of this system. 

## System Overview

The system collects images through a camera using OpenCV, annotates them using LabelMe, and then processes it through a machine learning pipeline with TensorFlow. Image augmentaiton is applied to enchance the dataset's quality through Albumentations. Finally, the deep learning model is built and trained using TensorFlow's Functional API. The model is exported for real time face detaction. 

## Prerequisites

To run this project, you need to have Python installed on your machine along with the following libraries:
- `labelme`
- `tensorflow`
- `opencv-python`
- `matplotlib`
- `albumentations`

You can install these packages using pip:
```bash
pip install labelme tensorflow opencv-python matplotlib albumentations
```

## Dataset Preparation
1. **Image Collection**: Use OpenCV to capture images that will be used for testing
2. **Anotation**: Annotate the images using LabelMe for creating precise models
3. **Review and Load Dataset**: Utilize TensorFlow, NumPy, and Matplotlib to inspect the dataset and implement an image loading function
4. **Data Splitting**: Manually split the dataset into training and testing sets to prepare for model training

## Model Building

Utilize the Functional API of TensorFlow to construct a deep learning model. Import the necessary layers and use VGG16 as a base neural network model as a face recognition system. A custom model classs is then created to manage the training process and monitor performance. To ensure the model is trained effectivley, validate its accuracy and loss metrics. 

## Prediction and Real Time Detection

After the creation of the model, load the trained model to make predictions on a test set and integrate it into a real time detection system using the output model. 
