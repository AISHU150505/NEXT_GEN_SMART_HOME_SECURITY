import os
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
from PIL import Image

# Paths
TEST_DIR = "frames_test"
MODEL_PATH = "models/resnet50_finetuned.pth"

# Device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Fine-Tuned ResNet50
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

# Classify Videos
print("[INFO] Classifying video frames...")
video_results = {}

for video_folder in os.listdir(TEST_DIR):
    video_path = os.path.join(TEST_DIR, video_folder)

    if os.path.isdir(video_path):
        print(f"\n[INFO] Processing video: {video_folder}")
        suspicious_count = 0
        total_frames = 0

        for img_name in sorted(os.listdir(video_path)):  # Sort to maintain frame order
            img_path = os.path.join(video_path, img_name)
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

            except:
                print(f"[WARNING] Skipping corrupted image: {img_path}")

        final_label = "Suspicious" if suspicious_count / total_frames > 0.6 else "Not Suspicious"

        video_results[video_folder] = {
            "Total Frames": total_frames,
            "Suspicious Frames": suspicious_count,
            "Not Suspicious Frames": total_frames - suspicious_count,
            "Final Decision": final_label
        }

print("\n==================== FINAL VIDEO CLASSIFICATIONS ====================")
for video, stats in video_results.items():
    print(f"\nVideo: {video}")
    print(f"  ➤ Total Frames: {stats['Total Frames']}")
    print(f"  ➤ Suspicious Frames: {stats['Suspicious Frames']}")
    print(f"  ➤ Not Suspicious Frames: {stats['Not Suspicious Frames']}")
    print(f"  ➤ 🔥 Final Decision: {stats['Final Decision']}")