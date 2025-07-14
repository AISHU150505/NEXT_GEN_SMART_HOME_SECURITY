import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import DatasetFolder
from torch.utils.data import DataLoader
from PIL import Image

# Paths
TRAIN_DIR = "frames_train"
MODEL_PATH = "models/resnet50_finetuned.pth"

# Hyperparameters
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.001

# Device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# Image Transformations
transform = transforms.Compose([
    # 🔹 Data Augmentation Steps (Applied Only During Training)
    transforms.RandomRotation(degrees=10),  # Small rotations (-10° to +10°)
    transforms.RandomHorizontalFlip(p=0.5),  # 50% chance of flipping image
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Adjust brightness/contrast
    transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),  # Random crops with 90%-100% size
    
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
])

# Function to Load Images from All Video Folders
def load_all_images(main_folder, label):
    """Loads all images from nested video folders and assigns a label."""
    all_images = []
    all_labels = []
    
    for video_folder in os.listdir(main_folder):  # Loop through each video folder
        video_path = os.path.join(main_folder, video_folder)
        if os.path.isdir(video_path):  # Ensure it's a directory
            for img_name in sorted(os.listdir(video_path)):  # Sort to maintain frame order
                img_path = os.path.join(video_path, img_name)
                try:
                    img = Image.open(img_path).convert("RGB")  # Load as RGB
                    all_images.append(transform(img))  # Apply transforms
                    all_labels.append(label)  # Assign label
                except:
                    print(f"[WARNING] Skipping corrupted image: {img_path}")
    
    return all_images, all_labels

# Load Suspicious & Not Suspicious Data
print("[INFO] Loading training dataset...")
suspicious_images, suspicious_labels = load_all_images(os.path.join(TRAIN_DIR, "suspicious"), label=1)
not_suspicious_images, not_suspicious_labels = load_all_images(os.path.join(TRAIN_DIR, "not_suspicious"), label=0)

# Convert to Tensors
X_train = torch.stack(suspicious_images + not_suspicious_images)  # Stack images into tensor
y_train = torch.tensor(suspicious_labels + not_suspicious_labels)  # Convert labels to tensor

# Create DataLoader
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Load Pretrained ResNet50
print("[INFO] Loading ResNet50...")
model = models.resnet50(pretrained=True)

# Modify Classifier for 2 Classes (Suspicious & Not Suspicious)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)  

# Freeze Early Layers (Train only last layers)
for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True

# Move Model to Device
model = model.to(device)

# Loss Function & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

# Training Loop
print("[INFO] Training ResNet50...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    accuracy = 100 * correct / total
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss:.4f}, Accuracy: {accuracy:.2f}%")

# Save Fine-Tuned Model
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), MODEL_PATH)
print(f"[INFO] Model saved at {MODEL_PATH}")