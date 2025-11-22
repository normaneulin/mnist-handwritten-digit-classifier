# MNIST Digit Classifier - Deep Learning Web Application

A production-ready handwritten digit classification system using Convolutional Neural Networks (CNNs) deployed as an interactive Streamlit web application.

**Status**: Fully Trained & Deployed  
**Framework**: TensorFlow/Keras + Streamlit  
**Best Model Accuracy**: 98.17% (Baseline CNN)  
**Last Updated**: November 22, 2025

---

## Project Structure

```
DL-Classifier-Web-Application/
├── colab/
│   ├── Deep_Learning_Model.ipynb          # Complete training notebook (all 4 models)
│   ├── baseline_cnn.keras                 # Best model (98.17% accuracy)
│   ├── dnn_dropout.keras                  # DNN model (94.91% accuracy)
│   ├── mobilenetv2_transfer.keras         # MobileNetV2 (63.17% accuracy)
│   ├── resnet50_transfer.keras            # ResNet50 (78.01% accuracy)
│   └── class_labels.json                  # Label mappings
│
├── dataset/
│   ├── mnist_train.csv                    # Training data (60,000 samples)
│   └── mnist_test.csv                     # Test data (10,000 samples)
│
├── app.py                                 # Streamlit web application
├── requirements.txt                       # Python dependencies
├── README.md                              # This file
└── export_all_models.py                   # Script to export all 4 models

# Additional files in root (copies of best models):
├── baseline_cnn.keras                     # Best - Baseline CNN
├── dnn_dropout.keras                      # DNN with Dropout
├── mobilenetv2_transfer.keras             # MobileNetV2 Transfer Learning
├── resnet50_transfer.keras                # ResNet50 Transfer Learning
└── class_labels.json                      # Class labels (0-9)
```

---

## Quick Start (for new users)

### Prerequisites
- **Python 3.11+** (recommended Python 3.13)
- **Windows/Mac/Linux**
- **Git** (for cloning)
- **~200MB disk space** (for dependencies + models)

### Step 1: Clone Repository and Navigate

```bash
git clone <repository-url>
cd DL-Classifier-Web-Application
```

### Step 2: Create Python Virtual Environment

**On Windows (PowerShell):**
```powershell
# Create virtual environment
python -m venv env

# Activate it
.\env\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

**On Mac/Linux (Bash):**
```bash
# Create virtual environment
python3 -m venv env

# Activate it
source env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

> **Note on Windows Path Limits**: If you encounter path length errors during installation, create the venv at a shorter path:
> ```powershell
> mkdir C:\projects\dlenv
> python -m venv C:\projects\dlenv
> C:\projects\dlenv\Scripts\Activate.ps1
> ```

### Step 3: Launch the Web Application

```bash
streamlit run app.py
```

The app will open automatically at: http://localhost:8501

---

## How to Use the Application

### Basic Workflow
1. **Upload Image**: Click upload area or drag-drop a handwritten digit image
   - Supported formats: `.jpg`, `.jpeg`, `.png`
   - Works best with clear, centered digits on white paper
2. **Preview**: See how the model sees your image (28×28 grayscale)
3. **Predict**: Click "Predict Digit" button
4. **Results**: View top-1 prediction (GREEN highlight) and top-5 predictions

### Tips for Best Results
- Use high contrast (dark ink on white paper)
- Center the digit in the image
- Ensure digit is clear and not blurry
- Upload square-ish images (~28x28 aspect ratio)
- Avoid rotated or tilted digits

---

## Available Models

You have 4 trained models to choose from. Switch between them by editing line 100 in app.py:

```python
model_path = "baseline_cnn.keras"  # Change this line
```

| Model Name | File | Size | Architecture | Accuracy | Best For |
|------------|------|------|--------------|----------|----------|
| Baseline CNN | baseline_cnn.keras | 0.65 MB | Custom CNN (54K params) | 98.17% | Production |
| DNN Dropout | dnn_dropout.keras | 2.84 MB | Dense NN (244K params) | 94.91% | Testing |
| MobileNetV2 | mobilenetv2_transfer.keras | 11.07 MB | Transfer Learning (2.4M params) | 63.17% | Research |
| ResNet50 | resnet50_transfer.keras | 93.63 MB | Transfer Learning (23.8M params) | 78.01% | Research |

### How to Switch Models

**Example: Using DNN model**
```python
# In app.py, line 100, change:
model_path = "dnn_dropout.keras"
```

The app will reload automatically and use the new model.

---

## Model Training (Optional)

If you want to **retrain the models** from scratch:

1. **Open the notebook:**
   ```bash
   jupyter notebook colab/Deep_Learning_Model.ipynb
   ```

2. **Run all cells** (or use Jupyter to run individual cells)
   - Cells 1-9: Data loading and preprocessing
   - Cells 10-13: Model architecture definitions
   - Cells 14-17: Model training
   - Cells 18-20: Model evaluation
   - Cell 21: Export all 4 models

3. **Models will be exported** to `colab/` folder

4. **Copy to root** (optional):
   ```bash
   copy colab/*.keras .
   copy colab/class_labels.json .
   ```

---

## Technical Details

### Model Architecture

**Baseline CNN (Best Performer - 98.17%)**
- Input: 28×28×1 grayscale images
- Conv2D: 32 filters, 3×3 kernel
- MaxPooling: 2×2
- Conv2D: 64 filters, 3×3 kernel
- MaxPooling: 2×2
- Flatten → Dense(128) → Dense(10)
- Activation: ReLU, Softmax (output)
- Total params: 54,410

Other Models:
- DNN: Flatten input -> Dense layers with Dropout
- MobileNetV2: Pre-trained ImageNet weights + custom top layers
- ResNet50: Pre-trained ImageNet weights + custom top layers

### Image Preprocessing Pipeline

```python
1. Convert to grayscale (PIL)
2. Resize to 28×28 (LANCZOS interpolation)
3. Normalize pixel values (0-1 range)
4. Auto-invert if needed (light background detection)
5. Reshape to (1, 28, 28, 1) for model input
```

**Key Feature**: Automatic image inversion detects if uploaded image has light background and inverts it to match MNIST training format (dark background with light digits).

### Training Dataset

- MNIST Dataset: 70,000 handwritten digit images (0-9)
  - Training: 60,000 images
  - Testing: 10,000 images
  - Resolution: 28x28 grayscale
  - Format: CSV with pixel values (0-255)

- Train/Validation Split: 80% train / 20% validation
- Augmentation: ImageDataGenerator with rotation, shift, zoom
- Epochs: Early stopping with patience=5
- Batch Size: 32

---

## Dependencies

All dependencies are in requirements.txt:

```
streamlit>=1.28.0          # Web framework
tensorflow>=2.13.0         # Deep learning
keras>=3.0.0              # Neural networks
numpy>=1.24.0             # Numerical computing
pandas>=2.0.0             # Data processing
pillow>=10.0.0            # Image handling
scikit-learn>=1.3.0       # ML utilities
matplotlib>=3.8.0         # Plotting
seaborn>=0.12.0           # Statistical plots
```

### Installation Issues?

Issue: ModuleNotFoundError when running app
```bash
# Make sure to activate your virtual environment first
.\env\Scripts\Activate.ps1

# Then install again
pip install -r requirements.txt
```

Issue: TensorFlow installation fails on Windows
```bash
# Use shorter path for venv (see Quick Start Step 2)
# Or try: pip install tensorflow-cpu (if GPU not needed)
```

---

## Features

User-Friendly Interface
- Drag-and-drop image upload
- Real-time preprocessing preview
- Beautiful sidebar with instructions

Smart Predictions
- Top-1 prediction highlighted in GREEN
- Top-5 predictions with confidence scores
- Visual confidence distribution chart
- Automatic image inversion detection

Production Ready
- Model caching for fast predictions
- Error handling and validation
- Support for multiple image formats
- Responsive design (desktop & mobile)

---

## Performance Metrics

Baseline CNN (Best Model)
```
Test Accuracy:  98.17%
Loss:           0.0549
Parameters:     54,410
File Size:      0.65 MB
Inference:      <100ms per image
```

Training History
- Baseline CNN: 16 epochs (early stopped)
- DNN: 5 epochs
- MobileNetV2: 5 epochs
- ResNet50: 5 epochs

---

## Troubleshooting

### App won't start
```bash
# Check Python version
python --version  # Should be 3.11+

# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Run with verbose output
streamlit run app.py --logger.level=debug
```

### Model not found error
```bash
# Ensure .keras files are in the same directory as app.py
# Check file exists:
ls *.keras

# If missing, copy from colab folder:
cp colab/*.keras .
```

### Predictions are wrong
- Ensure image has high contrast (dark digit on white)
- Try centered, square-ish images
- The app auto-inverts light backgrounds, but manual optimization helps

### Port 8501 already in use
```bash
# Use different port
streamlit run app.py --server.port 8502
```

---

## Deployment Options

### Local Deployment (Done)
```bash
streamlit run app.py
```

### Cloud Deployment (Streamlit Cloud)
1. Push repo to GitHub
2. Go to share.streamlit.io
3. Connect GitHub repo
4. Deploy with one click

### Docker Deployment
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit", "run", "app.py"]
```

---

## Learning Resources

- **Streamlit Docs**: https://docs.streamlit.io
- **TensorFlow Docs**: https://www.tensorflow.org/guide
- **Keras API**: https://keras.io
- **MNIST Dataset**: http://yann.lecun.com/exdb/mnist/
- **CNN Concepts**: https://cs231n.github.io

---

## Project Workflow Summary

```
┌─────────────────┐
│  1. Data Load   │ - 60K training + 10K test images
└────────┬────────┘
         │
┌────────▼──────────┐
│  2. Preprocess    │ - Normalize, reshape, encode
└────────┬──────────┘
         │
┌────────▼──────────┐
│  3. Train Models  │ - 4 architectures trained
└────────┬──────────┘
         │
┌────────▼──────────┐
│  4. Evaluate      │ - Best: Baseline CNN (98.17%)
└────────┬──────────┘
         │
┌────────▼──────────┐
│  5. Export        │ - Save to .keras files
└────────┬──────────┘
         │
┌────────▼──────────┐
│  6. Deploy        │ - Streamlit web app
└────────┬──────────┘
         │
┌────────▼──────────┐
│  7. Test & Use    │ - http://localhost:8501
└────────────────────┘
```

---

## Features Showcase

### Web App Capabilities
- Upload handwritten digit images
- Get instant predictions (98.17% accurate)
- View top-5 predictions with confidence
- See visual confidence distribution
- Preview preprocessed 28x28 image
- Mobile-responsive design
- Fast inference (<100ms)

### Notebook Capabilities
- Experiment with 4 different architectures
- Compare model performance
- Visualize training curves
- Retrain with new data
- Export models in Keras format

---

## Contributing

Feel free to:
- Add new model architectures
- Improve preprocessing
- Enhance UI/UX
- Fix bugs
- Add new features

---

## License

This project is provided for educational purposes.

---

## Support

For issues or questions:
1. Check Troubleshooting section above
2. Review notebook comments for implementation details
3. Check TensorFlow/Streamlit official docs

---

**Status**: Ready for Use
**Last Tested**: November 22, 2025
**Framework**: Streamlit + TensorFlow/Keras
**Python Version**: 3.11+

Happy Predicting!
