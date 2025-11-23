# MNIST Digit Classifier - Deep Learning Web Application

A handwritten digit classification system using Deep Neural Networks (DNN) deployed as an interactive Streamlit web application.

**Status**: Ready to Use  
**Framework**: TensorFlow/Keras + Streamlit  
**Last Updated**: November 23, 2025

---

## Project Structure

```
DL-Classifier-Web-Application/
├── app.py                                 # Streamlit web application
├── dnn_model.keras                        # Trained DNN model
├── class_labels.json                      # Class labels (0-9)
├── requirements.txt                       # Python dependencies
└── README.md                              # This file
```

---

## Quick Start (for new users)

### Prerequisites
- **Python 3.11+** (recommended Python 3.13)
- **Windows/Mac/Linux**
- **Git** (for cloning)
- **~100MB disk space** (for dependencies)

### Step 1: Clone Repository and Navigate

```bash
git clone <repository-url>
cd DL-Classifier-Web-Application
```

### Step 2: Create Python Virtual Environment

**On Windows (PowerShell) - Recommended (shorter path):**
```powershell
# Create virtual environment at a shorter path to avoid Windows path limits
mkdir C:\projects\dlenv
python -m venv C:\projects\dlenv

# Activate it
C:\projects\dlenv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

**On Windows (PowerShell) - Alternative (local env):**
```powershell
# Create virtual environment in project directory
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

> **Note on Windows Path Limits**: The recommended method creates venv at `C:\projects\dlenv` (shorter path) to avoid errors with long paths. If you prefer a local `env` folder, use the Alternative method above.

### Step 3: Launch the Web Application

**On Windows (PowerShell) - if using recommended `C:\projects\dlenv`:**
```powershell
# Activate your virtual environment
C:\projects\dlenv\Scripts\Activate.ps1

# Navigate to project directory
cd "path\to\DL-Classifier-Web-Application"

# Run the app
python -m streamlit run app.py
```

**On Windows (PowerShell) - if using local `env` folder:**
```powershell
# Activate your virtual environment
.\env\Scripts\Activate.ps1

# Run the app
python -m streamlit run app.py
```

**On Mac/Linux (Bash):**
```bash
# Activate your virtual environment
source env/bin/activate

# Run the app
python -m streamlit run app.py
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

## Technical Details

### Model Architecture

The application uses a Deep Neural Network (DNN) trained on the MNIST dataset for digit recognition.

### Image Preprocessing Pipeline

```python
1. Convert to grayscale (PIL)
2. Resize to 28×28 (LANCZOS interpolation)
3. Normalize pixel values (0-1 range)
4. Auto-invert if needed (light background detection)
5. Reshape to (1, 28, 28, 1) for model input
```

**Key Feature**: Automatic image inversion detects if uploaded image has light background and inverts it to match MNIST training format (dark background with light digits).

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
# Ensure dnn_model.keras file is in the same directory as app.py
# Check file exists:
ls *.keras
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

## Features Showcase

### Web App Capabilities
- Upload handwritten digit images
- Get instant predictions
- View top-5 predictions with confidence
- See visual confidence distribution
- Preview preprocessed 28x28 image
- Mobile-responsive design
- Fast inference

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

**Status**: Ready for Use
**Last Tested**: November 23, 2025
**Framework**: Streamlit + TensorFlow/Keras
**Python Version**: 3.11+