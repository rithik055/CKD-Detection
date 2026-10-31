# CKD Stage Prediction - Deep Learning Project

Deep learning system for Chronic Kidney Disease (CKD) stage prediction using CT scan segmentation and classification.

## 🎯 Project Overview

- **Segmentation**: U-Net model for kidney segmentation (Dice Score > 0.92)
- **Classification**: EfficientNet-B0 for CKD stage classification (4 stages)
- **Web Interface**: Flask-based UI for CT slice analysis

## 📁 Project Structure

```
ckdpred/
├── app.py                          # Main Flask application
├── show_progress.py                 # Training progress viewer
├── models_ckd/                     # Trained models
│   ├── unet_kidney_10k_30ep.pth
│   └── efficientnet_ckd_best_focal.pth
├── src/
│   ├── models/                     # Model architectures
│   ├── train/                      # Training scripts
│   │   └── train_ckd_classifier_focal.py  # Main training script
│   ├── preprocessing/              # Data preprocessing
│   ├── data/                       # Dataset classes
│   ├── inference/                  # Inference scripts
│   └── analysis/                   # Analysis tools
├── data/
│   ├── raw/                        # Raw CT slices and labels
│   ├── classification/             # ROI slices for classification
│   └── annotations/                # Segmentation masks
├── templates/                      # HTML templates
├── static/                         # CSS and results
└── logs/                           # Training logs

```

## 🚀 Quick Start

### 1. Run Web Application
```bash
python app.py
```
Access at: `http://127.0.0.1:5000`

### 2. Train Model
```bash
python src/train/train_ckd_classifier_focal.py
```

### 3. Monitor Training Progress
```bash
python show_progress.py
```

## 📊 Features

- **4 CKD Stages**: Well Preserved → Mild → Moderate → Advanced
- **Segmentation-led ROI**: U-Net extracts kidney regions
- **High Accuracy**: >96% validation accuracy
- **Professional UI**: Modern, accessible interface

## 🔧 Requirements

- Python 3.8+
- PyTorch
- Flask
- OpenCV
- Albumentations
- CUDA-capable GPU (recommended)

## 📝 Notes

- Training uses Focal Loss to address class bias
- Models are saved automatically during training
- Latest training logs are kept in `logs/` directory

