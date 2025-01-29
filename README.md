# Multi-Modal Deep Learning Model for Classification and Segmentation

## Overview
This project implements a multi-modal deep learning model leveraging a **ResNet-18 encoder** for **classification and segmentation** of medical images. The dataset used is the **COVID-19 Radiography Database**, which contains X-ray images useful for detecting and segmenting lung abnormalities associated with COVID-19.

## Dataset
- **Source**: https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database
- **Type**: Chest X-ray images (CXR)
- **Categories**:
  - COVID-19 positive cases
  - Normal cases
  - Viral Pneumonia
  - Lung Opacity

## Model Architecture
Here I summerized my metrics for classification and segmentation:
- **Classification**:
  - Backbone: `ResNet-18`
  - Optimizer: `Adam`
  - Loss Function: `CrossEntropyLoss`
  - Performance Metrics: `Accuracy`, `Classification Report`
- **Segmentation**:
  - Model: `UNet` with a `ResNet-18` encoder
  - Loss Function: `Dice Loss`
  - Performance Metrics: `Dice Score`

## Implementation Details
- **Libraries Used**:
  - `PyTorch`
  - `TorchMetrics`
  - `Segmentation-Models-PyTorch`
  - `Sci-kit Learn`
- **Training Process**:
  - Data loading and preprocessing (including label encoding)
  - Splitting into training and validation sets as well as resizing and making tensors
  - Training with learning rate of 1e-4 and `StepLR` scheduler with batch size of 32.
  - Evaluation using accuracy, classification report, and Dice Score
  - Checking the visualization for deeper comprehension of multi task model performance!
  (**A picture is worth a thousand words.**)
## Here are some of results of my model:
![](https://github.com/fajan-py/Multi-task-segmenter-and-classifier-of-CXR-/blob/main/download-1.png)
![](https://github.com/fajan-py/Multi-task-segmenter-and-classifier-of-CXR-/blob/main/download-2.png)
![](https://github.com/fajan-py/Multi-task-segmenter-and-classifier-of-CXR-/blob/main/download-3.png)
![](https://github.com/fajan-py/Multi-task-segmenter-and-classifier-of-CXR-/blob/main/download-4.png)
![](https://github.com/fajan-py/Multi-task-segmenter-and-classifier-of-CXR-/blob/main/download.png)
![](https://github.com/fajan-py/Multi-task-segmenter-and-classifier-of-CXR-/blob/main/Screen%20Shot%201403-11-10%20at%2011.42.56.png)




## Medical & Radiological Relevance
- **Automated COVID-19 Detection**: Can assist radiologists by providing a **computer-aided diagnosis (CAD) system** for detecting COVID-19 in chest X-rays.
- **Segmentation of Lung Regions**: Enables precise localization of **Lungs**, which can be helpful for further diagnosis of lung-region disease with help of other models.
- **Deep Learning in Radiology**: Demonstrates the effectiveness of CNN-based models for medical image analysis, reducing **human workload** and **enhancing diagnostic accuracy**.

## Results & Future Work
- **Performance**: Achieved **high classification accuracy** and **segmentation quality (Dice Score)**.


- **Potential Improvements**:
  - Integration of **multi-modal data** (e.g., CT scans) for improved diagnosis
  - Deployment as a **real-time radiological tool** in clinical settings

## How to Use
1. Install dependencies:
   ```bash
   pip install torchmetrics segmentation-models-pytorch

2. Load the dataset using Kaggle API.
3. Train the model by running the provided Jupyter Notebook.
4. Evaluate results using the classification and segmentation metrics.

## Tip: Just upload this notebook into your Colab space and run!
