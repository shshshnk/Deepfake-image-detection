
# Deepfake Image Detection Using Deep Learning

A deep learning project that classifies digital images as **Fake** or **Real** using three architectures: Basic CNN, ResNet50, and EfficientNet-B0.

## Overview

Deepfake technology can generate synthetic images that closely resemble genuine photographs. This project explores deep learning techniques for detecting fake images and compares the performance of different convolutional neural network architectures.

The project was implemented using Python and PyTorch in Google Colab with GPU acceleration.

## Objectives

- Develop a binary image classification system for fake and real images.
- Implement a Basic Convolutional Neural Network (CNN), ResNet50, and EfficientNet-B0.
- Apply transfer learning using pretrained models.
- Fine-tune ResNet50 to improve classification performance.
- Compare models using accuracy, precision, recall, F1-score, and confusion matrices.

## Dataset

**Dataset:** [Deepfake and Real Images — Kaggle](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images)

The dataset contains 190,335 images divided into training, validation, and test sets.

| Dataset Split | Fake Images | Real Images | Total |
|---|---:|---:|---:|
| Training | 70,001 | 70,001 | 140,002 |
| Validation | 19,641 | 19,787 | 39,428 |
| Testing | 5,492 | 5,413 | 10,905 |
| **Total** | **95,134** | **95,201** | **190,335** |

The dataset is organized into `Train`, `Validation`, and `Test` folders, each containing `Fake` and `Real` subfolders.

## Data Preprocessing

The following preprocessing steps were applied:

- Resize images to **224 × 224 pixels**.
- Convert images into PyTorch tensors.
- Normalize image channels using ImageNet mean and standard deviation values.
- Apply random horizontal flipping to training images.
- Use a batch size of 32.

## Models Implemented

### 1. Basic CNN
A custom convolutional neural network trained from scratch on the dataset.

### 2. ResNet50
A pretrained ResNet50 model adapted for two-class classification using transfer learning. Its final convolutional block and classification layer were subsequently fine-tuned.

### 3. EfficientNet-B0
A pretrained EfficientNet-B0 model adapted for two-class classification, with its feature-extraction layers frozen during the experiment.

## Results

The following test accuracies were recorded:

| Model Configuration | Test Accuracy |
|---|---:|
| Basic CNN | 79.45% |
| ResNet50 — Before Fine-Tuning | 72.31% |
| ResNet50 — After Fine-Tuning | **83.32%** |
| EfficientNet-B0 | 69.65% |

Fine-tuned ResNet50 achieved the highest test accuracy among the evaluated configurations.

### Fine-Tuned ResNet50: Classification Metrics

| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Fake | 80.93% | 87.49% | 84.08% |
| Real | 86.17% | 79.09% | 82.48% |

The overall test accuracy was **83.32%** on 10,905 test images.

## Technologies Used

- Python
- PyTorch
- torchvision
- NumPy
- scikit-learn
- Matplotlib
- Google Colab
- Kaggle

## Repository Structure

```text
deepfake-image-detection/
├── README.md
├── .gitignore
├── requirements.txt
├── notebooks/
├── results/
└── reports/
```

The folders will contain the actual notebooks, evaluation outputs, and report files as they are added to the repository.

## Limitations

- The models were evaluated on a single dataset.
- Some images were incorrectly classified.
- Performance on unseen image-generation methods has not been established.
- The project focuses on image classification, not video deepfake detection.
- A separate graphical user interface has not been implemented.

## Future Improvements

- Evaluate models on additional and more diverse datasets.
- Explore newer CNN and vision transformer architectures.
- Improve generalization to unseen image-generation techniques.
- Develop a graphical user interface for image uploads and predictions.
- Extend the approach to video deepfake detection.

## Contributors

- Shashank Kumar
- Harsh Rajpurohit
- Sumit Chaudhary

**Institution:** Karnavati University, Unitedworld Institute of Technology (UIT)  
**Program:** B.Tech CSE (AIML)  
**Course:** Deep Learning  
**Academic Year:** 2026–27

## Dataset Attribution

The dataset is provided through Kaggle: [Deepfake and Real Images Dataset](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images).
