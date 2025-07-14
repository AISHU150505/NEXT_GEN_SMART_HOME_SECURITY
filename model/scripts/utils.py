import os
import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms

# Load Pretrained CNN (ResNet50)
resnet50 = models.resnet50(pretrained=True)
resnet50 = torch.nn.Sequential(*list(resnet50.children())[:-1])  # Remove classification layer
resnet50.eval()  # Set to evaluation mode

# Define Preprocessing Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Rescale to 224x224
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
])

def extract_cnn_features(image):
    """Extracts deep features from an image using ResNet50."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    image_pil = transforms.ToPILImage()(image_rgb)  # Convert to PIL
    image_tensor = transform(image_pil).unsqueeze(0)  # Apply transforms

    with torch.no_grad():
        features = resnet50(image_tensor)  # Extract deep features
    return features.view(-1).numpy()  # Flatten feature vector

def load_images_from_folder(folder, max_images=500):
    """Loads images from a folder and returns a list of images."""
    images = []
    image_files = sorted(os.listdir(folder))[:max_images]  # Limit max images
    for filename in image_files:
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)  # Read image
        if img is not None:
            images.append(img)
    return images