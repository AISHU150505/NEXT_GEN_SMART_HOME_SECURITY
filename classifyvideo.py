import cv2
import motion_detector
import face_recognition_system1 as face_recognition1
import alert_system
import time
import os
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import shutil

# Paths
SUSPICIOUS_DIR = "recordings/suspicious_videos"
RECORDINGS_DIR = "recordings"
MODEL_PATH = r"C:\Users\pmkir\OneDrive\Desktop\project\final working project\model\models\resnet50_finetuned.pth"
LOG_FILE = "logs/security_log.txt"

# Ensure directories exist
os.makedirs(SUSPICIOUS_DIR, exist_ok=True)
os.makedirs(RECORDINGS_DIR, exist_ok=True)

# Load Fine-Tuned ResNet50 Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=False)
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, 2)  
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
model.to(device)

# Transformations for Images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
def log_event(message):
    """ Log security events to a file. """
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}\n"
    with open(LOG_FILE, "a") as log_file:
        log_file.write(log_entry)

def extract_frames(video_path, output_folder):
    """ Extract frames from a recorded video and save them as images. """
    cap = cv2.VideoCapture(video_path)
    count = 0
    os.makedirs(output_folder, exist_ok=True)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(output_folder, f"frame_{count:04d}.png")
        cv2.imwrite(frame_path, frame)
        count += 1

    cap.release()
    return output_folder  # Return the frame folder path

def classify_video(frame_folder, video_path):
    """ Classify frames and determine if the video is suspicious. """
    print(f"\n[INFO] Classifying video: {frame_folder}")
    log_event(f"Classifying video: {video_path}")
    suspicious_count = 0
    total_frames = 0

    for img_name in sorted(os.listdir(frame_folder)):  
        img_path = os.path.join(frame_folder, img_name)
        try:
            img = Image.open(img_path).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(img_tensor)
                _, predicted = torch.max(outputs, 1)

            label = "Suspicious" if predicted.item() == 1 else "Not Suspicious"
            print(f"Frame {total_frames + 1}: {label}")

            if predicted.item() == 1:
                suspicious_count += 1
            total_frames += 1

        except Exception as e:
            print(f"[WARNING] Skipping corrupted image: {img_path} ({str(e)})")

    # Decide if the video is suspicious
    final_label = "Suspicious" if suspicious_count / max(total_frames, 1) > 0.45 else "Not Suspicious"
    print(f"Classified as {final_label}")
    log_event(f"Classified as {final_label}")

    if final_label == "Suspicious":
        shutil.move(video_path, os.path.join(SUSPICIOUS_DIR, os.path.basename(video_path)))
        print(f"⚠️ Video moved to suspicious_videos!")
        log_event(f"⚠️ Video moved to suspicious_videos!")
    return final_label