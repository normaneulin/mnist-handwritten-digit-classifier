"""
Script to save the best model and class labels for deployment
Run this after training completes in the notebook
"""

import json
import os

def save_best_model(model, model_name="mnist_classifier.keras", labels_file="class_labels.json"):
    """
    Save the trained model and class labels
    
    Args:
        model: Trained Keras model
        model_name: Name to save the model as
        labels_file: JSON file to save class labels
    """
    # Save model
    model.save(model_name)
    print(f"✓ Model saved as {model_name}")
    
    # Save class labels (for MNIST: 0-9)
    labels = [str(i) for i in range(10)]
    with open(labels_file, 'w') as f:
        json.dump(labels, f)
    print(f"✓ Class labels saved as {labels_file}")
    print(f"Labels: {labels}")

# Usage in notebook:
# After identifying the best model, run:
# save_best_model(best_model, "mnist_classifier.keras", "class_labels.json")
