import torch
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import our custom classes
from utils.video_dataset import VideoFrameDataset
from models.cnn_rnn_detector import CnnRnnDetector

# ===================================================================
# CONFIGURATION
# ===================================================================
# Path to the dataset we will use for the final evaluation (our validation set)
TEST_DIR = "data_videos/val"

# Path to our best trained model from Phase 2
MODEL_PATH = "models/cnn_rnn_detector.pth"

# Parameters must match the training configuration
BATCH_SIZE = 8
SEQUENCE_LENGTH = 30

# ===================================================================

def evaluate_final_model():
    """
    Loads the trained CNN+RNN model and evaluates its performance on the test set.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Starting Final Model Evaluation on {device} ---")

    # --- 1. Load Model ---
    print(f"Loading model from: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"FATAL: Model file not found at {MODEL_PATH}")

    model = CnnRnnDetector().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval() # Set model to evaluation mode
    print("Model loaded successfully.")

    # --- 2. Load Test Data ---
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("Loading test dataset...")
    test_dataset = VideoFrameDataset(data_dir=TEST_DIR, transform=transform, sequence_length=SEQUENCE_LENGTH)
    if len(test_dataset) == 0:
        raise ValueError(f"No videos found in {TEST_DIR}. Please check the path.")

    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    print(f"Test dataset loaded. Found {len(test_dataset)} videos.")

    # --- 3. Run Predictions ---
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for video_sequences, labels in tqdm(test_loader, desc="Evaluating on Test Set"):
            video_sequences, labels = video_sequences.to(device), labels.to(device)
            outputs = model(video_sequences)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # --- 4. Generate and Display Results ---
    class_names = ['fake', 'real']
    
    # --- Classification Report ---
    print("\n" + "="*55)
    print("--- Final Classification Report ---")
    print("="*55)
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    print(report)

    # --- Confusion Matrix ---
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(pd.DataFrame(cm, index=class_names, columns=class_names), annot=True, fmt='d', cmap='Blues')
    plt.title('Final Confusion Matrix', fontsize=16)
    plt.ylabel('Actual Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    output_path = "evaluation_results/final_video_model_confusion_matrix.png"
    os.makedirs("evaluation_results", exist_ok=True)
    plt.savefig(output_path)
    print(f"\n✅ Confusion matrix plot saved to '{output_path}'")
    
if __name__ == '__main__':
    evaluate_final_model()

