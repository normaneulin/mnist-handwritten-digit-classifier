import numpy as np
from tensorflow import keras
from PIL import Image, ImageDraw
import json

# Load model
model = keras.models.load_model('baseline_cnn.keras')

# Load class labels
with open('class_labels.json', 'r') as f:
    labels = json.load(f)

print("=" * 80)
print("TESTING MNIST MODEL WITH DIFFERENT IMAGE FORMATS")
print("=" * 80)

# Test 1: Black on white (typical user upload)
print("\nTest 1: Black digit on WHITE background")
img_black_white = Image.new('L', (28, 28), color=255)  # White background
draw = ImageDraw.Draw(img_black_white)
draw.text((5, 5), "3", fill=0)  # Black digit

img_array = np.array(img_black_white) / 255.0
img_array = img_array.reshape(1, 28, 28, 1)
pred = model.predict(img_array, verbose=0)
pred_class = np.argmax(pred[0])
confidence = np.max(pred[0])
print(f"  Prediction: {pred_class} (confidence: {confidence*100:.2f}%)")
print(f"  Top 3: {[(labels[i], f'{pred[0][i]*100:.1f}%') for i in np.argsort(pred[0])[-1:-4:-1]]}")

# Test 2: White on black (MNIST format)
print("\nTest 2: WHITE digit on BLACK background")
img_white_black = Image.new('L', (28, 28), color=0)  # Black background
draw = ImageDraw.Draw(img_white_black)
draw.text((5, 5), "3", fill=255)  # White digit

img_array = np.array(img_white_black) / 255.0
img_array = img_array.reshape(1, 28, 28, 1)
pred = model.predict(img_array, verbose=0)
pred_class = np.argmax(pred[0])
confidence = np.max(pred[0])
print(f"  Prediction: {pred_class} (confidence: {confidence*100:.2f}%)")
print(f"  Top 3: {[(labels[i], f'{pred[0][i]*100:.1f}%') for i in np.argsort(pred[0])[-1:-4:-1]]}")

# Test 3: Inverted black on white
print("\nTest 3: Inverted (BLACK on WHITE → inverted to WHITE on BLACK)")
img_black_white_inv = Image.new('L', (28, 28), color=255)
draw = ImageDraw.Draw(img_black_white_inv)
draw.text((5, 5), "3", fill=0)

img_array = np.array(img_black_white_inv) / 255.0
# Invert: 1.0 - pixel_value
img_array = 1.0 - img_array
img_array = img_array.reshape(1, 28, 28, 1)
pred = model.predict(img_array, verbose=0)
pred_class = np.argmax(pred[0])
confidence = np.max(pred[0])
print(f"  Prediction: {pred_class} (confidence: {confidence*100:.2f}%)")
print(f"  Top 3: {[(labels[i], f'{pred[0][i]*100:.1f}%') for i in np.argsort(pred[0])[-1:-4:-1]]}")

print("\n" + "=" * 80)
print("ANALYSIS:")
print("=" * 80)
print("The model was trained on MNIST data: WHITE digits on BLACK background")
print("User uploads typically have: BLACK digits on WHITE background")
print("Solution: INVERT the image if background is light!")
print("=" * 80)
