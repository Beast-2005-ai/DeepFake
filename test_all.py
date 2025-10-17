import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings

# Suppress a common warning from sklearn when a class has no predictions
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

# ===================================================================
# IMPORT ALL YOUR MODEL CLASSES AND FUNCTIONS
# ===================================================================
from models.baseline_cnn import BaselineCNN
from models.resnet import ResNetNet
from models.xception import XceptionNet
from models.efficientnet import EfficientNetModel
from models.mesonet import MesoNet
from models.vit import build_vit_model
from models.adapgrnet import get_adapgrnet
from models.smdnet import get_smdnet


def get_model(model_name, device):
    """Loads a model instance and its corresponding trained weights."""
    model_path = os.path.join("models", f"{model_name}.pth")
    if not os.path.exists(model_path):
        print(f"Warning: Model weights not found at {model_path}. Skipping.")
        return None

    # Dictionary mapping model names to their initializers
    model_map = {
        'baseline_cnn': BaselineCNN(),
        'resnet': ResNetNet(pretrained=False), # Set pretrained=False as we load our own weights
        'xception': XceptionNet(),
        'efficientnet': EfficientNetModel(),
        'mesonet': MesoNet(),
        'vit': build_vit_model(),
        'adapgrnet': get_adapgrnet(),
        'smdnet': get_smdnet()
    }
    
    model = model_map.get(model_name)
    if model is None:
        print(f"Warning: Unknown model name '{model_name}'. Skipping.")
        return None

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        return model.to(device)
    except Exception as e:
        print(f"Error loading weights for '{model_name}': {e}. Skipping.")
        return None

def get_test_loader(model_name, data_dir="data", batch_size=32):
    """Returns a DataLoader with the correct transforms for the specified model."""
    transform_128 = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    transform_224 = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Select transform based on model
    transform = transform_128 if model_name in ['baseline_cnn', 'mesonet'] else transform_224
    
    test_path = os.path.join(data_dir, "test")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"FATAL: Test data directory not found at '{test_path}'")
        
    test_dataset = datasets.ImageFolder(root=test_path, transform=transform)
    # Use a smaller batch size for the large ViT model to avoid memory issues
    b_size = 16 if model_name == 'vit' else batch_size
    return DataLoader(test_dataset, batch_size=b_size, shuffle=False, num_workers=4), test_dataset.classes

def evaluate_and_compare():
    """Main function to evaluate all models and generate a comparison table."""
    model_names = [
        'baseline_cnn', 'resnet', 'xception', 'efficientnet',
        'mesonet', 'vit', 'adapgrnet', 'smdnet'
    ]
    
    all_metrics = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = "evaluation_results"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"--- Starting Full Model Evaluation on {device} ---")

    for model_name in model_names:
        print("\n" + "="*50)
        print(f"Evaluating: {model_name.upper()}")
        
        model = get_model(model_name, device)
        if model is None: continue

        try:
            test_loader, class_names = get_test_loader(model_name)
        except Exception as e:
            print(f"Error creating data loader for {model_name}: {e}. Skipping.")
            continue

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc=f"Testing {model_name}"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        if not all_labels:
            print("No labels found in the test set. Skipping metrics.")
            continue

        # --- Generate and Store Metrics ---
        report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True, zero_division=0)
        
        # Ensure 'fake' class exists in the report before accessing
        fake_metrics = report.get('fake', {'precision': 0, 'recall': 0})

        metrics = {
            "Model": model_name,
            "Accuracy": report['accuracy'],
            "F1-Score (Weighted)": report['weighted avg']['f1-score'],
            "Precision (Fake)": fake_metrics['precision'],
            "Recall (Fake)": fake_metrics['recall']
        }
        all_metrics.append(metrics)
        
        # --- Generate and Save Confusion Matrix ---
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(pd.DataFrame(cm, index=class_names, columns=class_names), annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix: {model_name.upper()}', fontsize=16)
        plt.ylabel('Actual Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.savefig(os.path.join(output_dir, f"{model_name}_confusion_matrix.png"))
        plt.close()

    if not all_metrics:
        print("\nNo models were successfully evaluated. Please check your .pth files and data paths.")
        return
        
    # --- Display Final Comparison Table ---
    df = pd.DataFrame(all_metrics).round(4)
    df.sort_values(by="F1-Score (Weighted)", ascending=False, inplace=True)
    df.set_index("Model", inplace=True)
    
    print("\n\n" + "="*70)
    print("--- FINAL MODEL PERFORMANCE COMPARISON ---")
    print("="*70)
    print(df.to_string())
    print("="*70)
    print(f"\n✅ Evaluation complete. All plots saved in '{output_dir}/' folder.")

if __name__ == "__main__":
    evaluate_and_compare()