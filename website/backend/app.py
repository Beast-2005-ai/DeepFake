import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import torch
import cv2
from torchvision import transforms
import uuid

# --- Import Your Model Architectures ---
from models.smdnet import get_smdnet
from models.cnn_rnn_detector import CnnRnnDetector

# --- App Initialization ---
app = Flask(__name__)
CORS(app)

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_MODEL_PATH = "models/smdnet_finetuned.pth"
VIDEO_MODEL_PATH = "models/cnn_rnn_detector.pth"
TEMP_FOLDER = "temp_uploads"
os.makedirs(TEMP_FOLDER, exist_ok=True)

# --- Load Models ---
print("Loading models into memory...")

# Load models using the method that matches your standalone scripts
image_model = get_smdnet(num_classes=2).to(DEVICE)
image_model.load_state_dict(torch.load(IMAGE_MODEL_PATH, map_location=DEVICE))
image_model.eval()

video_model = CnnRnnDetector().to(DEVICE)
video_model.load_state_dict(torch.load(VIDEO_MODEL_PATH, map_location=DEVICE))
video_model.eval()

print("✅ Models loaded successfully.")

# --- Define Transformations ---
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
video_frame_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
CLASS_NAMES = ['fake', 'real']

# ===================================================================
# API ENDPOINTS
# ===================================================================

@app.route("/predict/image", methods=["POST"])
def predict_image():
    if 'files' not in request.files:
        return jsonify({"error": "No files part in the request"}), 400

    files = request.files.getlist('files')
    results = []
    
    with torch.no_grad():
        for file in files:
            temp_filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
            temp_path = os.path.join(TEMP_FOLDER, temp_filename)
            file.save(temp_path)
            
            try:
                image = Image.open(temp_path).convert("RGB")
                tensor = image_transform(image).unsqueeze(0).to(DEVICE)
                output = image_model(tensor)
                _, pred_idx = torch.max(output, 1)
                prediction = CLASS_NAMES[pred_idx.item()]
                results.append({"filename": file.filename, "prediction": prediction})
            except Exception as e:
                print(f"Error processing image {file.filename}: {e}")
                results.append({"filename": file.filename, "prediction": "Error"})
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

    return jsonify(results)


@app.route("/predict/video", methods=["POST"])
def predict_video():
    if 'files' not in request.files:
        return jsonify({"error": "No files part in the request"}), 400

    files = request.files.getlist('files')
    results = []

    with torch.no_grad():
        for file in files:
            temp_filename = str(uuid.uuid4()) + ".mp4"
            video_path = os.path.join(TEMP_FOLDER, temp_filename)
            file.save(video_path)
            
            prediction = "Error" # Default prediction

            try:
                frames = []
                cap = cv2.VideoCapture(video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                seq_length = 30
                
                if total_frames < seq_length: seq_length = total_frames
                
                start_frame = (total_frames - seq_length) // 2
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

                for _ in range(seq_length):
                    ret, frame = cap.read()
                    if not ret: break
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    frames.append(video_frame_transform(pil_image))
                
                cap.release()

                if len(frames) > 0:
                    frames_tensor = torch.stack(frames).unsqueeze(0).to(DEVICE)
                    output = video_model(frames_tensor)
                    _, pred_idx = torch.max(output.data, 1)
                    prediction = CLASS_NAMES[pred_idx.item()]
                else:
                    prediction = "Error: Could not read frames"
            except Exception as e:
                print(f"Error processing video {file.filename}: {e}")
                prediction = "Error"
            finally:
                if os.path.exists(video_path):
                    os.remove(video_path)
            
            results.append({"filename": file.filename, "prediction": prediction})
            
    return jsonify(results)

# --- Main Execution ---
if __name__ == "__main__":
    app.run(debug=True, port=5000)

