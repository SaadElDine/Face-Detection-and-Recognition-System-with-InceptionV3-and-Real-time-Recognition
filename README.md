# Face-Detection-and-Recognition-System-with-InceptionV3-and-Real-time-Recognition

## Introduction

This project aims to develop a face recognition system using deep learning techniques. The project consists of three notebooks, each focusing on different aspects of the face recognition pipeline. The notebooks are:

1. **Face Detection and Cropping**: This notebook focuses on detecting faces in images using OpenCV and cropping them for further processing.
2. **Model Training**: This notebook is dedicated to training a deep learning model for face recognition using the InceptionV3 architecture.
3. **Real-time Face Recognition**: This notebook demonstrates real-time face recognition using a webcam, utilizing the trained model and embeddings extracted during training.

## 1. Face Detection and Cropping Notebook

### Overview

This notebook aims to extract a dataset of images containing faces and process them for face detection and cropping. It uses a pre-trained face detection model to detect faces in images and saves the cropped faces for further processing.

### Key Steps

1. **Loading the Dataset**: The notebook loads a dataset of images containing faces. This dataset is used for face detection and cropping.
2. **Face Detection**: It loads a pre-trained face detection model using OpenCV. This model is used to detect faces in the images from the dataset.
3. **Face Cropping**: Detected faces are cropped from the images and saved in a separate directory for further processing.

## 2. Model Training Notebook

### Overview

This notebook focuses on training a deep learning model for face recognition using the InceptionV3 architecture. It fine-tunes the pre-trained InceptionV3 model on the extracted face images and prepares it for face recognition.

### Key Steps

1. **Data Preprocessing**: The extracted face images are preprocessed for training. This includes resizing and normalizing the images.
2. **Model Architecture**: It loads the pre-trained InceptionV3 model and freezes its layers. Additional layers are added for classification.
3. **Model Training**: The model is compiled and trained using the preprocessed face images.
4. **Transfer Learning**: The model is trained using transfer learning, where the pre-trained InceptionV3 model is fine-tuned on the face dataset.
5. **Model Compilation** The model is compiled for face recognition.
6. **Model Saving**: Once training is complete, the model is saved for future use.

## 3. Real-time Face Recognition Notebook

### Overview

This notebook demonstrates real-time face recognition using a webcam. It loads the trained face recognition model and class embeddings extracted during training. The notebook processes webcam frames for face recognition and displays the recognized faces in real-time.

### Key Steps

1. **Loading the Model**: The pre-trained face recognition model is loaded from the saved file.
2. **Loading Class Embeddings**: Class embeddings for each class (person) in the dataset are loaded from the saved file.
3. **Face Recognition**: Webcam frames are processed for face recognition. The model predicts embeddings for each detected face, which are compared with the class embeddings using cosine similarity.
4. **Cosine Similarity**: Cosine similarity is used to measure the similarity between embeddings. If the similarity exceeds a predefined threshold, the face is recognized as belonging to a known person.
5. **Real-time Display**: Recognized faces are displayed in real-time on the webcam feed.

### Test Cases

![elonmask](https://github.com/SaadElDine/Face-Detection-and-Recognition-System-with-InceptionV3-and-Real-time-Recognition/assets/113860522/c61fde95-6f22-430c-bc3b-6d22463064ed)

![megan fox](https://github.com/SaadElDine/Face-Detection-and-Recognition-System-with-InceptionV3-and-Real-time-Recognition/assets/113860522/222ae2ab-aa3b-405d-a8ff-bc034b159740)

![me](https://github.com/SaadElDine/Face-Detection-and-Recognition-System-with-InceptionV3-and-Real-time-Recognition/assets/113860522/a75c74fc-2334-4619-8354-a0df0c4fae2a)

## Conclusion

Face recognition project demonstrates a comprehensive approach to building a face recognition system using deep learning. By combining face detection, model training, and real-time recognition, the project showcases a practical application of deep learning in computer vision.
