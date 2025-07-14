import argparse
import os

def train_model():
    print("[INFO] Training model...")
    os.system("python scripts/train_model.py")

def classify_video():
    print("[INFO] Classifying video...")
    os.system("python scripts/classify_video.py")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robbery Detection System")
    parser.add_argument("--train", action="store_true", help="Train the SVM model")
    parser.add_argument("--classify", action="store_true", help="Classify frames from test videos")

    args = parser.parse_args()

    if args.train:
        train_model()
    elif args.classify:
        classify_video()
    else:
        print("[ERROR] No argument provided. Use --train or --classify")