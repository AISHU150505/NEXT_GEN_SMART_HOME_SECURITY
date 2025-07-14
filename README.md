# 🔐 Next-Gen Smart Home Security System
A real-time surveillance solution integrating facial recognition and suspicious activity detection using deep learning. This project offers a proactive approach to smart home security through intelligent video analytics, real-time alerts, and an interactive web interface.

## 📌 Overview
This system leverages a fine-tuned **ResNet-50** model for classifying live video feeds as suspicious or normal and identifying individuals using facial recognition. It uses **modular Python scripts**, **multithreading**, and a **Flask-based web interface** for seamless surveillance and management.

## 🚀 Features
- 👁️ Real-time motion-triggered video surveillance
- 🧠 Facial recognition with known/unknown classification
- ⚠️ Suspicious activity detection (e.g., robbery, vandalism)
- 📢 Audio & console alerts for unknown or suspicious entities
- 🌐 Web-based interface for monitoring and visitor management
- 🧩 Modular codebase and scalable architecture

## 🏗️ System Architecture
```
[Camera Feed] → [Motion Detection] → [Face Recognition & Activity Classification (ResNet-50)] → [Alert System + Video Logging] → [Flask Web Interface]
```

## 🛠️ Tech Stack
- **Languages**: Python
- **Libraries**: OpenCV, face_recognition, pyttsx3, PyTorch, Flask
- **Model**: ResNet-50 (fine-tuned on UCF-Crime dataset)
- **Hardware**: NVIDIA RTX 3080, Intel i5, 32GB RAM
- **Platform**: Local (Windows 11), with future cloud integration possible

## 🧩 Project Structure
```
.
├── Main.py                    # Entry point - orchestrates detection, classification, alerts
├── Motion_detector.py        # Detects motion in live video feed
├── Face_encoding.py          # Encodes faces into vector embeddings
├── Face_recognition_system.py# Identifies known/unknown faces
├── Train_model.py            # Trains ResNet-50 on suspicious activity
├── Classify_video.py         # Runs trained model on recorded footage
├── Alert_system.py           # Triggers sound/console alerts
├── App.py                    # Flask backend for UI
├── /templates/               # HTML templates for UI
├── /logs/                    # Logs of detected events
├── /dataset/                 # Known/unknown faces, suspicious clips
└── requirements.txt          # Python dependencies
```

## 🧪 Dataset
- **UCF-Crime Dataset Subset**
  - 5 suspicious classes: Robbery, Burglary, Vandalism, Shoplifting, Stealing
  - 1 normal class
  - Frame-based binary classification: suspicious vs. normal
  - Data preprocessed at 1 FPS, 224x224, ImageNet normalized

## 📊 Performance
| Metric        | Value              |
|---------------|--------------------|
| Accuracy      | 93.4% (Avg)         |
| Inference Time| 30–35 ms/frame     |
| FPS           | ~30 (1080p)        |
| False Pos Rate| < 4% across classes|

## 🧠 Model Comparison
| Method        | Accuracy | Inference | FPR   |
|---------------|----------|-----------|--------|
| LSTM-based    | 85%      | ~100 ms   | 10%    |
| CNN-based     | 88%      | ~50 ms    | 8%     |
| **ResNet-50** | **93.4%**| **30 ms** | **2.5%**|

## 🧪 Example Use Cases
- Real-time detection of strangers at doorstep
- Intrusion or break-in alerts with video proof
- Visitor log management via web dashboard

## ⚠️ Limitations
- Limited to entrance surveillance
- No mobile app/cloud integration (yet)
- Performance may degrade in low-light conditions

## 📈 Future Enhancements
- Integration with smart locks and IoT devices
- Cloud storage and mobile push notifications
- Outdoor support with adaptive lighting handling
- Expansion to multi-camera setup

## ✅ Getting Started

### 🔧 Installation
1. Clone the repo
```bash
git clone https://github.com/AISHU150505/NEXT_GEN_SMART_HOME_SECURITY.git
cd next-gen-home-security
```

2. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

### ▶️ Run the System
```bash
python Main.py
```

### 🌐 Launch the Web Interface
```bash
python App.py
```

Then navigate to `http://127.0.0.1:5000` in your browser.

---

## 👥 Authors
- **Aishwarya S**  
- **Kiruthiga P M**  
- **Christy Shawn Franco** 
## 📄 License
This project is licensed for academic and research use only.

## 📚 References
- UCF-Crime Dataset: [https://www.crcv.ucf.edu/research/projects/ucf-crime](https://www.crcv.ucf.edu/research/projects/ucf-crime)
- Related Work: IJMTST, Electronics, IDCIoT, ICDICI papers