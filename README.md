# 🫁 Pneumonia Detection from Chest X-Rays

A deep learning project that detects pneumonia from chest X-ray images using CNNs, transfer learning (ResNet50, DenseNet121), and an R-CNN model for localization. Built as a capstone project on the RSNA Pneumonia Detection Challenge dataset.

---

## 📌 Motivation

Pneumonia is one of the leading causes of death worldwide, particularly in children under 5 and the elderly. Timely diagnosis is critical, but access to expert radiologists is limited in many parts of the world. This project explores how computer vision can assist in automating pneumonia detection from chest X-rays, potentially making diagnosis faster, cheaper, and more accessible.

---

## 📂 Dataset

**Source:** [RSNA Pneumonia Detection Challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge) (Kaggle)

| Property | Details |
|---|---|
| Total patients | 26,684 |
| Pneumonia cases | 6,012 (22.5%) |
| Normal cases | 20,672 (77.5%) |
| Image format | DICOM (.dcm) |
| Image size | 1024 × 1024 px (resized to 224×224 for training) |

Key observations from EDA:
- Significant **class imbalance** — addressed using class weights during training
- Pneumonia most prevalent in the **50–60 age group**, with most cases between ages 30–70
- **Males** were more affected (~2,177 cases) compared to females (~1,497 cases)

---

## 🧠 Models & Results

Eight models were trained and evaluated, progressively increasing in complexity:

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Base CNN | 0.6281 | 0.4417 | 0.5886 | 0.5047 |
| CNN + L2 Regularization | 0.7281 | 0.5722 | 0.6158 | 0.5932 |
| CNN + Data Augmentation | 0.7175 | 0.5806 | 0.4414 | 0.5015 |
| Transfer Learning (continued training) | 0.7313 | 0.3992 | 0.7071 | 0.5103 |
| ResNet50 | 0.7595 | 0.4689 | 0.5141 | 0.4905 |
| Fine-Tuned ResNet50 | 0.7595 | 0.5806 | 0.5141 | 0.5341 |
| DenseNet121 | 0.7344 | 0.4505 | 0.8186 | 0.5812 |
| Fine-Tuned DenseNet121 | 0.7344 | 0.4505 | 0.8186 | 0.5812 |
| **R-CNN (DenseNet backbone)** | **0.7544** | **0.4705** | **0.8186** | **0.6012** |

> **Note on metric choice:** In a medical diagnosis context, **recall** is the most critical metric — a false negative (missed pneumonia) is far more dangerous than a false positive. The DenseNet and R-CNN models achieved the highest recall of 0.8186.

---

## 🔧 Pipeline

```
DICOM Images
     │
     ▼
Image Preprocessing (resize to 224×224, normalize, convert to RGB)
     │
     ▼
Data Augmentation (Albumentations — flips, rotations, brightness)
     │
     ▼
Class Weight Balancing (to handle 77/23 class imbalance)
     │
     ▼
Model Training (Base CNN → Transfer Learning → ResNet50 → DenseNet → R-CNN)
     │
     ▼
Evaluation (Accuracy, Precision, Recall, F1, Confusion Matrix)
```

---

## 🛠️ Tech Stack

- **Python 3** · TensorFlow / Keras · PyTorch (Mask R-CNN attempt)
- **Libraries:** NumPy, Pandas, Matplotlib, Seaborn, Plotly, Scikit-learn
- **Image processing:** pydicom, Albumentations, OpenCV
- **Environment:** Google Colab (GPU)

---

## 📁 Repository Structure

```
Pneumonia-Detection/
├── Capstone_Final_Code.ipynb   # Full notebook with code, outputs & analysis
├── Capstone_Final_Code.html    # HTML site with code, outputs & analysis
├── Capstone_Final_Presentation.pptx     # Presentation slides
└── README.md
```

---

## 📊 Key Findings

- Transfer learning on medical images significantly outperforms a scratch-trained CNN
- **DenseNet121** is well-suited for this task due to its dense connectivity and feature reuse
- The **R-CNN model** (DenseNet backbone + bounding box head) achieved the best F1 score by combining classification with localization
- The Mask R-CNN experiment was limited by Colab's compute constraints but showed promising direction

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/ananthananthananth/pneumonia-detection-cnn.git
   ```
2. Open `Capstone_Final_Code.ipynb` in Google Colab
3. Mount your Google Drive and update the dataset paths in the notebook
4. Download the dataset from [Kaggle](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge) and place it in your Drive

---

*This project was completed as a capstone submission. Dataset courtesy of the Radiological Society of North America (RSNA).*
