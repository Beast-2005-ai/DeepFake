import torch
import cv2
import os
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# Import our custom model class
from models.cnn_rnn_detector import CnnRnnDetector

def predict_video_folder(input_folder, model_path, sequence_length=30):
    """
    Loads the trained video model and predicts on all video files in a folder.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Load Model ---
    model = CnnRnnDetector().to(device)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"FATAL: Model file not found at {model_path}")
    # Added weights_only=True to address the FutureWarning
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print("Model loaded successfully.")

    # --- 2. Define Transformations ---
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    class_names = ['fake', 'real']

    # --- 3. Find all video files in the folder ---
    video_files = [f for f in os.listdir(input_folder) if f.endswith('.mp4')]
    if not video_files:
        print(f"No .mp4 video files found in the specified folder: {input_folder}")
        return

    print(f"\nFound {len(video_files)} videos to classify...")
    all_results = []

    # --- 4. Loop through each video and predict ---
    with torch.no_grad():
        for video_file in tqdm(video_files, desc="Classifying Videos"):
            video_path = os.path.join(input_folder, video_file)
            frames = []
            
            try:
                cap = cv2.VideoCapture(video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                if total_frames < sequence_length:
                    print(f"\nWarning: Video '{video_file}' is too short. Using all {total_frames} frames.")
                    current_seq_len = total_frames
                else:
                    current_seq_len = sequence_length

                start_frame = (total_frames - current_seq_len) // 2
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                
                for _ in range(current_seq_len):
                    ret, frame = cap.read()
                    if not ret: break
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    frames.append(transform(pil_image))
                
                cap.release()

                if len(frames) > 0:
                    frames_tensor = torch.stack(frames).unsqueeze(0).to(device)
                    output = model(frames_tensor)
                    _, predicted_idx = torch.max(output.data, 1)
                    prediction = class_names[predicted_idx.item()]
                    all_results.append({'file': video_file, 'prediction': prediction})
                else:
                    all_results.append({'file': video_file, 'prediction': 'ERROR: Could not read frames'})

            except Exception as e:
                print(f"\nError processing {video_file}: {e}")
                all_results.append({'file': video_file, 'prediction': 'ERROR'})

    return all_results

if __name__ == '__main__':
    # ===================================================================
    # EDIT THIS LINE to point to your folder of videos for testing
    # ===================================================================
    VIDEO_FOLDER_PATH = "C:/Users/Sujal/Downloads/test_videos"

    # This path points to your best trained model
    MODEL_PATH = "models/cnn_rnn_detector.pth"
    
    # Run the prediction process on the folder
    predictions = predict_video_folder(VIDEO_FOLDER_PATH, MODEL_PATH)
    
    # --- 5. Display All Results ---
    if predictions:
        print("\n" + "="*50)
        print("--- FINAL CLASSIFICATION RESULTS ---")
        for res in predictions:
            print(f"  File: {res['file']:<30} | Prediction: {res['prediction'].upper()}")
        print("="*50)
