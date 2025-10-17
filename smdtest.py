import torch
import os
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from models.smdnet import get_smdnet

def predict_with_finetuned_model(input_folder, model_path):
    """
    Loads the fine-tuned SMDNet model and classifies all images in a given folder.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = get_smdnet(num_classes=2)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"FATAL: The specified model weights were not found at '{model_path}'.")

    print(f"Loading fine-tuned model weights from: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    class_names = ['fake', 'real']

    supported_extensions = ['.jpg', '.jpeg', '.png']
    image_files = [f for f in os.listdir(input_folder) if os.path.splitext(f)[1].lower() in supported_extensions]
    
    if not image_files:
        print(f"No supported images (.jpg, .jpeg, .png) found in the folder: {input_folder}")
        return

    results = []
    print(f"\nFound {len(image_files)} images to classify...")

    with torch.no_grad():
        for filename in tqdm(image_files, desc="Classifying Images"):
            try:
                image_path = os.path.join(input_folder, filename)
                image = Image.open(image_path).convert('RGB')
                input_tensor = transform(image).unsqueeze(0)
                input_tensor = input_tensor.to(device)
                output = model(input_tensor)
                _, predicted_idx = torch.max(output, 1)
                prediction = class_names[predicted_idx.item()]
                results.append({'file': filename, 'prediction': prediction})
            except Exception as e:
                print(f"\nCould not process {filename}. Error: {e}")
    
    return results

if __name__ == '__main__':
    IMAGE_FOLDER_PATH = "C:/Users/Sujal/DeepFake/testing"
    MODEL_PATH = "models/smdnet_finetuned.pth"
    predictions = predict_with_finetuned_model(IMAGE_FOLDER_PATH, MODEL_PATH)
    
    if predictions:
        print("\n" + "="*45)
        print("--- CLASSIFICATION RESULTS ---")
        for res in predictions:
            print(f"  File: {res['file']:<30} | Prediction: {res['prediction'].upper()}")
        print("="*45)
