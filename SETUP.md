# QUICK SETUP GUIDE

For developers who want to quickly get the MNIST Digit Classifier running.

## TL;DR (Too Long; Didn't Read)

```bash
# Clone repo
git clone <repo-url>
cd DL-Classifier-Web-Application

# Setup environment
python -m venv env
.\env\Scripts\Activate.ps1           # Windows
source env/bin/activate             # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app.py
```

**Done!** Open http://localhost:8501

---

## Step-by-Step Guide

### Prerequisites
- Python 3.11+ installed
- Git installed
- ~200MB free disk space

### Clone Repository
```bash
git clone <repository-url>
cd DL-Classifier-Web-Application
```

### Create Virtual Environment

Windows (PowerShell):
```powershell
python -m venv env
.\env\Scripts\Activate.ps1
```

Mac/Linux (Bash):
```bash
python3 -m venv env
source env/bin/activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

Time required: 3-5 minutes (first time installation of TensorFlow)

### Launch Application
```bash
streamlit run app.py
```

Your browser will open to: http://localhost:8501

---

## Use the Application

1. **Upload Image**: Click the upload area or drag a digit image
2. **Preview**: See the 28×28 preprocessed image
3. **Predict**: Click "Predict Digit"
4. **Results**: View top-1 (green) and top-5 predictions

---

## 🔄 Switch Between Models

Edit line 100 in `app.py`:

```python
# Current (Baseline CNN - 98.17%):
model_path = "baseline_cnn.keras"

# Or try:
model_path = "dnn_dropout.keras"                # 94.91%
model_path = "mobilenetv2_transfer.keras"       # 63.17%
model_path = "resnet50_transfer.keras"          # 78.01%
```

Save and the app reloads automatically!

---

## Troubleshooting

**Python not found?**
```bash
# Try:
py -m venv env           # Windows
python3 -m venv env      # Mac/Linux
```

**Pip install fails?**
```bash
# Upgrade pip first:
python -m pip install --upgrade pip

# Then try again:
pip install -r requirements.txt
```

**Port 8501 already in use?**
```bash
streamlit run app.py --server.port 8502
```

**Model file not found?**
```bash
# Make sure you're in the correct directory:
cd DL-Classifier-Web-Application

# Check files exist:
ls *.keras
ls class_labels.json
```

---

## 📚 Full Documentation

See **README.md** for:
- Complete feature list
- Model architecture details
- Training information
- Deployment options
- Technical details

---

## Want to Retrain Models?

First, extract the dataset:

**Windows (PowerShell):**
```powershell
Expand-Archive -Path dataset.zip -DestinationPath dataset/
```

**Mac/Linux (Bash):**
```bash
unzip dataset.zip -d dataset/
```

Then open and run the Jupyter notebook:
```bash
jupyter notebook colab/Deep_Learning_Model.ipynb
```

Then run all cells to train 4 different models from scratch.

---

## Checklist

- [ ] Git cloned the repo
- [ ] Python 3.11+ installed
- [ ] Virtual environment created and activated
- [ ] Dependencies installed (pip install -r requirements.txt)
- [ ] App launched (streamlit run app.py)
- [ ] Browser opened to http://localhost:8501
- [ ] Uploaded a test image
- [ ] Got a prediction!

All done? Great!

---

## Still Having Issues?

1. Check the Troubleshooting section above
2. Read the README.md file
3. Check TensorFlow/Streamlit official documentation
4. Open an issue on GitHub

---

Happy Digit Classifying!
