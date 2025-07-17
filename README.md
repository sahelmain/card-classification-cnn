# Card Classification using Convolutional Neural Networks (CNNs)

> **CS 5388 Project**: Automated playing card recognition using deep learning approaches

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19+-orange)](https://tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## 🎯 Project Overview

This project implements and compares two CNN approaches for classifying standard playing cards from images:

1. **Custom Sequential CNN** - Built from scratch with convolutional and pooling layers
2. **Transfer Learning** - Fine-tuned pre-trained models (EfficientNet-B0, MobileNetV2)

**Dataset**: 8,154 high-resolution playing card images across 53 classes (including jokers)
- 📂 **Training**: 7,624 images
- 📂 **Validation**: 265 images  
- 📂 **Testing**: 265 images

## 🚀 Quick Results

| Model | Test Accuracy | Training Time | Parameters |
|-------|--------------|---------------|------------|
| **Custom CNN** | **81.4%** | ~45 min | 134M |
| **Lightweight CNN** | **74.0%** | 3.2 min | 4.4M |
| **MobileNetV2** | 51.7% | 2.4 min | 2.8M |

*Note: Training times on Apple Silicon Mac with Metal GPU acceleration*

## 🏗️ Project Structure

```
card-classification/
├── 📁 data/                    # Dataset (train/valid/test splits)
├── 📁 src/                     # Source code
│   ├── data_preprocessing.py   # Data loading and preprocessing
│   ├── custom_cnn.py          # Custom CNN implementation
│   ├── efficientnet_model.py  # EfficientNet transfer learning
│   ├── quick_demo.py          # Fast training demo
│   ├── gpu_config.py          # GPU/Metal acceleration setup
│   ├── verify_dataset.py      # Dataset structure verification
│   └── evaluate.py            # Model evaluation
├── 📁 results/                # Trained models (.h5 files)
├── 📁 docs/                   # Documentation and reports
├── 📄 requirements.txt        # Python dependencies
└── 📄 README.md              # This file
```

## 🛠️ Setup & Installation

### Prerequisites
- Python 3.8+
- TensorFlow 2.0+
- 8GB+ RAM recommended

### Installation
```bash
# Clone the repository
git clone <your-repo-url>
cd card-classification

# Install dependencies
pip install -r requirements.txt

# Verify dataset structure
python src/verify_dataset.py
```

### Dataset Setup
1. Download the [Cards Image Dataset](https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification) from Kaggle
2. Extract to the `data/` folder with structure:
   ```
   data/
   ├── train/ (53 card class folders)
   ├── valid/ (53 card class folders)
   └── test/  (53 card class folders)
   ```

## 🎮 Usage

### Quick Demo (Recommended)
```bash
# Fast training with lightweight models (~5-10 minutes)
python src/quick_demo.py
```

### Full Training
```bash
# Train custom CNN from scratch
python src/custom_cnn.py

# Train EfficientNet with transfer learning
python src/efficientnet_model.py

# Evaluate all models
python src/evaluate.py
```

## 📊 Methodology

### Custom CNN Architecture
- **Conv2D layers**: Progressive feature extraction (32→64→128 filters)
- **MaxPooling**: Spatial dimensionality reduction
- **Dropout**: Regularization (0.5 rate)
- **Dense layers**: Classification head (512→53 units)

### Transfer Learning Approach
- **Base Model**: EfficientNet-B0 pre-trained on ImageNet
- **Fine-tuning**: Frozen base + custom classifier head
- **Progressive training**: Initial training → fine-tuning last layers

### Optimizations
- ✅ **GPU Acceleration**: Apple Metal / CUDA support
- ✅ **Early Stopping**: Prevents overfitting
- ✅ **Learning Rate Scheduling**: Adaptive learning rates
- ✅ **Data Augmentation**: Image preprocessing pipelines

## 🔬 Key Findings

1. **Custom CNNs** achieved higher accuracy but required more training time
2. **Transfer learning** provided faster convergence but lower peak accuracy  
3. **Apple Metal GPU** acceleration significantly reduced training time on Mac
4. **Data quality** and preprocessing were crucial for model performance

## 🏆 Applications

- 🎰 **Casino surveillance** - Automated card detection
- 🎮 **AR gaming** - Real-time card recognition
- 🤖 **Robotic sorting** - Automated card handling systems
- 📚 **Educational tools** - Interactive card learning apps

## 📈 Future Improvements

- [ ] Implement data augmentation for better generalization
- [ ] Test with more transfer learning architectures (ResNet, Vision Transformer)
- [ ] Add real-time inference capabilities
- [ ] Deploy as web application or mobile app
- [ ] Expand to other card games (UNO, Tarot, etc.)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for improvements.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Baris Ozcan** - Texas Tech University
- **Nameera Khan** - Texas Tech University  
- **Namra Khan** - Texas Tech University
- **Sahel Azzam** - Texas Tech University

---

> **Note**: This implementation demonstrates practical deep learning techniques for computer vision tasks and serves as a benchmark for card classification research. 