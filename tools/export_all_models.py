"""
Export all 4 trained models from the notebook to individual .keras files
Run this script after training all models in the notebook
"""

import json
import os
from tensorflow import keras

# Model information
MODELS_INFO = {
    "baseline_model": {
        "filename": "baseline_cnn.keras",
        "name": "Baseline CNN",
        "description": "Custom CNN architecture optimized for MNIST",
        "params": 54410,
        "accuracy": 0.9817
    },
    "dnn_model": {
        "filename": "dnn_dropout.keras",
        "name": "DNN with Dropout",
        "description": "Dense Neural Network with dropout regularization",
        "params": 244522,
        "accuracy": 0.9491
    },
    "mobilenetv2_model": {
        "filename": "mobilenetv2_transfer.keras",
        "name": "MobileNetV2 Transfer Learning",
        "description": "Pre-trained MobileNetV2 with custom top layers",
        "params": 2423242,
        "accuracy": 0.6317
    },
    "resnet50_model": {
        "filename": "resnet50_transfer.keras",
        "name": "ResNet50 Transfer Learning",
        "description": "Pre-trained ResNet50 with custom top layers",
        "params": 23851274,
        "accuracy": 0.7801
    }
}

class_labels = [str(i) for i in range(10)]

def export_models_from_notebook():
    """
    Export models from notebook kernel variables.
    This function is designed to be run in the notebook after training.
    """
    print("=" * 80)
    print("EXPORTING ALL TRAINED MODELS")
    print("=" * 80)
    
    # Get the notebook kernel
    try:
        from IPython import get_ipython
        ipython = get_ipython()
        user_ns = ipython.user_ns
    except:
        print("❌ This script must be run in a Jupyter notebook!")
        return False
    
    exported_models = []
    
    for model_var, info in MODELS_INFO.items():
        print(f"\n{'─' * 80}")
        print(f"Exporting: {info['name']}")
        print(f"{'─' * 80}")
        
        # Get model from notebook kernel
        if model_var not in user_ns:
            print(f"❌ Model '{model_var}' not found in notebook kernel")
            print(f"   Make sure you've trained all models in the notebook first!")
            continue
        
        model = user_ns[model_var]
        filename = info['filename']
        
        try:
            # Save model
            model.save(filename)
            file_size = os.path.getsize(filename) / (1024 * 1024)  # Convert to MB
            
            print(f"✓ Model saved: {filename}")
            print(f"  - File size: {file_size:.2f} MB")
            print(f"  - Architecture: {info['description']}")
            print(f"  - Parameters: {info['params']:,}")
            print(f"  - Test Accuracy: {info['accuracy']*100:.2f}%")
            
            exported_models.append({
                "filename": filename,
                "name": info['name'],
                "accuracy": info['accuracy']
            })
            
        except Exception as e:
            print(f"❌ Error saving {filename}: {e}")
    
    # Save class labels
    print(f"\n{'─' * 80}")
    print("Saving class labels")
    print(f"{'─' * 80}")
    
    with open('class_labels.json', 'w') as f:
        json.dump(class_labels, f)
    
    print(f"✓ Class labels saved: class_labels.json")
    
    # Create summary file
    summary = {
        "models": exported_models,
        "class_labels": class_labels,
        "total_models": len(exported_models)
    }
    
    with open('models_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Summary saved: models_summary.json")
    
    # Print final summary
    print(f"\n{'=' * 80}")
    print("EXPORT SUMMARY")
    print(f"{'=' * 80}")
    print(f"✓ Total models exported: {len(exported_models)}")
    for model_info in exported_models:
        print(f"  - {model_info['name']}: {model_info['filename']} ({model_info['accuracy']*100:.2f}% accuracy)")
    print(f"\n✓ All files ready for web deployment!")
    print(f"{'=' * 80}")
    
    return True

if __name__ == "__main__":
    print("\n🚀 RUNNING IN NOTEBOOK MODE")
    print("Add this to a cell in your notebook to export all models:")
    print("\n" + "─" * 80)
    print("from export_all_models import export_models_from_notebook")
    print("export_models_from_notebook()")
    print("─" * 80 + "\n")
