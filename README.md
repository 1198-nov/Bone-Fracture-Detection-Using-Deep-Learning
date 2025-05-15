# 🦴 Bone Fracture Detection Using Deep Learning

This project explores the use of deep learning models for the automated detection of bone fractures in X-ray images. By leveraging advanced convolutional neural networks (CNNs) and transfer learning techniques, we aim to support radiologists with quick, accurate, and objective diagnostic assistance—especially in resource-limited environments.

---

## 🧠 Introduction

Bone fracture detection is vital for ensuring timely and accurate treatment. Misdiagnosis or delayed diagnosis can lead to severe complications. Manual review of X-rays is time-consuming and subject to human error, especially in high-stress environments or locations with limited radiology professionals.

This project showcases how deep learning models can automate this task efficiently and improve healthcare accessibility.

---

## 🎯 Project Objective

- Automate the detection of bone fractures from X-ray images.
- Compare performance between a custom CNN and pretrained models (ResNet-18, VGG16, DenseNet121, EfficientNet-B0).
- Reduce diagnostic errors and assist radiologists in clinical decision-making.

---

## 🗂️ Dataset

- **Source**: ResearchGate and Kaggle
- **Categories**: Elbow, Fingers, Forearm, Humerus, Shoulder, Wrist
- **Split**:
  - Training: 1807 images
  - Validation: 173 images
  - Test: 83 images
- **Challenges**:
  - Class imbalance across different fracture types
  - Limited test set size

---

## ⚙️ Methodology

### 1. **Data Preprocessing**
- Resizing images to 224x224
- Normalization
- Augmentation: Rotation, flipping, scaling
- Filtering techniques: CLAHE, bilateral filter, median filtering

### 2. **Model Training**
- Custom CNN: Built from scratch with dropout, batch normalization, and ReLU activation
- Transfer Learning:
  - **ResNet-18**
  - **VGG16**
  - **DenseNet-121**
  - **EfficientNet-B0**

### 3. **Evaluation Metrics**
- Accuracy
- Loss (training vs validation)
- Confusion Matrix
- PSNR (Peak Signal-to-Noise Ratio) for image quality post-processing

---

## 🏗️ Model Architecture

### 🔹 Custom CNN
- 4 Convolutional layers with BatchNorm, ReLU, and Dropout (p=0.3)
- MaxPooling layers
- Fully connected layers with Kaiming initialization and L2 regularization

### 🔹 Transfer Learning
- Final layers of all pretrained models replaced to fit 6-class classification
- Fine-tuned on fracture X-ray dataset

---

## 📊 Performance

| Model           | Accuracy |
|----------------|----------|
| Custom CNN     | 50%      |
| VGG16          | 89%      |
| ResNet-18      | 90%      |
| DenseNet-121   | 86%      |
| EfficientNet-B0| 89%      |

- **ResNet-18** yielded the best accuracy with 90%.
- **Confusion matrix** and PSNR analysis provided insights into model confidence and image enhancement effects.

---

## 🧩 Challenges & Improvements

### Challenges:
- Overfitting due to small dataset
- Class imbalance
- High computational cost for training deeper models

### Improvements:
- Hyperparameter tuning
- More data augmentation and advanced filtering
- Leveraging additional pretrained models

---

## 🌍 Project Impact

This project highlights the capability of AI to:
- Assist in early diagnosis and reduce treatment delays
- Serve as a second opinion for radiologists
- Improve patient care in emergency and rural settings

It also answers critical research questions in medical AI:
1. Can transfer learning help with small medical datasets?
2. Which CNN architectures perform best for fracture detection?
3. How do preprocessing techniques improve model accuracy?

---

## 🚀 Future Work

With additional resources, we plan to:
- ✅ Add **object detection/segmentation** for precise localization of fractures
- ✅ Implement **fracture severity grading**
- ✅ Expand coverage to more bones (e.g., hip, skull, spine)
- ✅ Incorporate **multi-modal data** (CT, MRI, clinical records)

---

## 🧪 How to Run

```bash
# Clone the repository
git clone https://github.com/1198-nov/Bone-Fracture-Detection-Using-Deep-Learning
cd bone-fracture-detection

# Install dependencies
pip install -r requirements.txt

# Run the training script
python train_model.py

# Evaluate on test set
python evaluate_model.py
