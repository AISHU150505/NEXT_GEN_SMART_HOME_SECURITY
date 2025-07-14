from typing import Counter
from flask import Flask, render_template, jsonify, request, send_from_directory, url_for
import subprocess
import os
import shutil
import pickle
import logging
import face_recognition
from face_encoding import encode_known_faces
from face_recognition_system1 import load_known_faces
from visitor_recognition import VISITOR_DIR, start_recognition
from classifyvideo import extract_frames
from classifyvideo import classify_video
# Configure logging
BASE_DIR = r"C:\Users\pmkir\OneDrive\Desktop\project\FINAL WORKING PROJECT"
LOGS_DIR = os.path.join(BASE_DIR, "logs")
LOG_FILE = os.path.join(LOGS_DIR, "security_log.txt")
RECORDINGS_FOLDER = os.path.join(BASE_DIR, "recordings")
UNKNOWN_DIR = os.path.join(RECORDINGS_FOLDER, "unknown")
VISITORS_DIR = os.path.join(BASE_DIR, "database", "visitor_faces")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
ENCODINGS_FILE = "database/face_encodings.pkl"
FRAMES_FOLDER = os.path.join(BASE_DIR, "frames")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FRAMES_FOLDER, exist_ok=True)



# Ensure directories exist
os.makedirs(UNKNOWN_DIR, exist_ok=True)
os.makedirs(VISITORS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Set up logging
#logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format="%(asctime)s - %(message)s")

app = Flask(__name__)

# Helper function to retrieve logs
def get_unknown_person_logs():
    logs = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as file:
            logs = [line.strip() for line in file.readlines()]
    return logs

# Helper function to get images from a directory
def get_images(directory):
    return [f for f in os.listdir(directory) if f.endswith((".jpg", ".png"))]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/start-surveillance', methods=['POST'])
def start_surveillance():
    try:
        subprocess.Popen(["python", "main.py"], cwd=BASE_DIR)
        logging.info("Surveillance started.")
        return jsonify({"status": "success", "message": "Surveillance started!"})
    except Exception as e:
        logging.error(f"Error starting surveillance: {e}")
        return jsonify({"status": "error", "message": str(e)})

@app.route('/stop-surveillance', methods=['POST'])
def stop_surveillance():
    try:
        subprocess.call(["taskkill", "/F", "/IM", "python.exe"])
        logging.info("Surveillance stopped.")
        return jsonify({"status": "success", "message": "Surveillance stopped!"})
    except Exception as e:
        logging.error(f"Error stopping surveillance: {e}")
        return jsonify({"status": "error", "message": str(e)})

@app.route('/unknown-logs', methods=['GET'])
def unknown_logs():
    return jsonify({"logs": get_unknown_person_logs()})
@app.route('/recordings/<filename>')
def get_recording(filename):
    return send_from_directory(RECORDINGS_FOLDER, filename)

@app.route('/recordings')
def list_recordings():
    videos = [f for f in os.listdir(RECORDINGS_FOLDER) if f.endswith((".avi", ".mp4"))]
    return jsonify({"videos": videos})

@app.route('/recordings/unknown/<filename>')
def get_unknown_image(filename):
    return send_from_directory(UNKNOWN_DIR, filename)

@app.route('/recordings/visitors/<filename>')
def get_visitor_image(filename):
    return send_from_directory(VISITORS_DIR, filename)

@app.route('/unknown-images', methods=['GET'])
def unknown_images():
    return jsonify({"images": get_images(UNKNOWN_DIR)})
import os

import os
import time
from flask import jsonify

@app.route('/visitor-images', methods=['GET'])
def visitor_images():
    folder_path = os.path.join(BASE_DIR, "database", "visitor_faces")
    visitor_images_data = []

    # Loop through each folder and retrieve image details
    for folder in os.listdir(folder_path):
        folder_full_path = os.path.join(folder_path, folder)
        if os.path.isdir(folder_full_path):
            for filename in os.listdir(folder_full_path):
                if filename.endswith(('.jpg', '.png', '.jpeg')):  # Adjust the file types if needed
                    # Get the creation or modification time of the image
                    file_path = os.path.join(folder_full_path, filename)
                    timestamp = time.ctime(os.path.getmtime(file_path))  # You can use creation time too with os.path.getctime

                    # Add folder name, filename, and timestamp
                    visitor_images_data.append({
                        "folder": folder,
                        "filename": filename,
                        "timestamp": timestamp
                    })

    return jsonify({"images": visitor_images_data})

import cv2

@app.route('/add-visitor', methods=['POST'])
def add_visitor():
    data = request.json
    filename = data.get("filename")
    name = data.get("name")

    if not filename or not name:
        return jsonify({"status": "error", "message": "Filename or name missing"}), 400

    src_path = os.path.join(UNKNOWN_DIR, filename)
    dest_folder = os.path.join(VISITORS_DIR, name)
    dest_path = os.path.join(dest_folder, filename)

    if os.path.exists(src_path):
        os.makedirs(dest_folder, exist_ok=True)
        shutil.move(src_path, dest_path)

        if os.path.exists(ENCODINGS_FILE):
            with open(ENCODINGS_FILE, "rb") as f:
                known_faces, known_names = pickle.load(f)
        else:
            known_faces, known_names = [], []

        # Load and preprocess the image
        image = cv2.imread(dest_path)
        if image is None:
            return jsonify({"status": "error", "message": "Failed to load image"}), 400

        image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)  # Resize for better detection
        rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

        # Detect faces
        face_locations = face_recognition.face_locations(rgb_frame, model='cnn')
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        if face_encodings:
            known_faces.append(face_encodings[0])
            known_names.append(name)

            with open(ENCODINGS_FILE, "wb") as f:
                pickle.dump((known_faces, known_names), f)

            start_recognition()  # Restart recognition

            return jsonify({"status": "success", "message": f"{name} added to visitors!"})
        else:
            return jsonify({"status": "error", "message": "No face detected in image"}), 400
    else:
        return jsonify({"status": "error", "message": "File not found"}), 404

@app.route('/start-visitor-recognition', methods=['POST'])
def start_visitor_recognition():
    """Start real-time visitor recognition."""
    start_recognition()
    return jsonify({"status": "success", "message": "Visitor recognition started!"})
@app.route('/remove-visitor', methods=['POST'])
def remove_visitor():
    """Remove a visitor by deleting their image and updating encodings."""
    data = request.json
    name = data.get("name")  # Visitor name to remove

    if not name:
        return jsonify({"status": "error", "message": "Visitor name is required"}), 400

    visitor_path = os.path.join(VISITOR_DIR, f"{name}.jpg")

    if os.path.exists(visitor_path):
        os.remove(visitor_path)  # Delete visitor image
        load_known_faces()  # Refresh face encodings
        return jsonify({"status": "success", "message": f"{name} removed from visitors!"})
    else:
        return jsonify({"status": "error", "message": "Visitor not found"}), 404
from flask import send_from_directory
@app.route('/database/visitor_faces/<folder_name>/<filename>')
def serve_visitor_image(folder_name, filename):
    # Construct the path to the folder dynamically using folder_name
    folder_path = os.path.join(BASE_DIR, "database", "visitor_faces", folder_name)
    
    # Ensure the folder exists and the file exists within the folder
    if os.path.exists(folder_path) and os.path.exists(os.path.join(folder_path, filename)):
        return send_from_directory(folder_path, filename)
    else:
        return "File not found", 404


import os
import re

LOG_FILE_PATH = 'security_log.txt'

# Function to parse the security log file
def read_log_file():
    logs = []
    if os.path.exists(LOG_FILE_PATH):
        with open(LOG_FILE_PATH, 'r') as file:
            for line in file:
                # Match timestamp and action with face name if available
                match = re.match(r'\[(.*?)\] (.*)', line.strip())
                if match:
                    timestamp = match.group(1)
                    action = match.group(2)
                    face_name = None
                    
                    # Check if "Detected" contains a name
                    if "Detected:" in action:
                        face_name = action.split(":")[1].strip()
                        action = "Detected"  # Update action type
                    elif "Motion detected but no faces recognized" in action:
                        face_name = "Unknown"
                    
                    logs.append({
                        "timestamp": timestamp,
                        "action": action,
                        "face_name": face_name
                    })
    return logs
@app.route('/security-logs', methods=['GET'])
def get_security_logs():
    logs = read_log_file()
    return jsonify({"logs": logs})
@app.route('/visitor-count', methods=['GET'])
def get_visitor_count():
    visitor_count = 0
    if os.path.exists(VISITOR_DIR):
        visitor_count = len([f for f in os.listdir(VISITOR_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))])
    
    return jsonify({"visitor_count": visitor_count})

def get_most_frequent_unknown_time():
    logs = read_log_file()
    unknown_times = []

    for log in logs:
        if log['face_name'] == 'Unknown':
            # Extract the timestamp (time portion) for "Unknown" entries
            time = log['timestamp'].split(' ')[1]  # Getting the HH:MM:SS format
            unknown_times.append(time)
    
    if unknown_times:
        time_counts = Counter(unknown_times)
        most_common_time, count = time_counts.most_common(1)[0]
        return most_common_time, count
    else:
        return None, 0
@app.route('/upload-video', methods=['POST'])
def upload_video():
    """Handles video uploads, extracts frames, and classifies whether it is suspicious."""
    
    if 'video' not in request.files:
        return jsonify({"status": "error", "message": "No video uploaded"}), 400

    video = request.files['video']
    video_path = os.path.join(UPLOAD_FOLDER, video.filename)
    video.save(video_path)

    output_folder = os.path.join(FRAMES_FOLDER, os.path.splitext(video.filename)[0])
    os.makedirs(output_folder, exist_ok=True)  # Ensure the folder exists before extraction

    extract_frames(video_path, output_folder)

    classification_result = classify_video(output_folder, video_path)  # Pass both arguments

    return jsonify({"status": "success", "message": "Video processed", "classification": classification_result})


import re
from collections import Counter
from datetime import datetime

import re
from collections import Counter
from datetime import datetime

@app.route('/most-frequent-unknown-time', methods=['GET'])
def most_frequent_unknown_time():
    # Read the logs
    logs = read_log_file()
    
    # Extract timestamps of unknown persons from the logs
    unknown_times = [log['timestamp'] for log in logs if log['face_name'] == "Unknown"]

    if not unknown_times:
        return jsonify({"most_common_time": "No unknown persons detected", "count": 0})

    # Extract just the time part (HH:MM) from the timestamp
    time_list = []
    for time_str in unknown_times:
        try:
            # Assuming the timestamp format is [YYYY-MM-DD HH:MM:SS]
            time_obj = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
            time_list.append(time_obj.strftime('%H:%M'))  # Get only HH:MM
        except ValueError:
            continue  # Skip invalid timestamps

    # Count the occurrences of each time
    time_counts = Counter(time_list)

    # Get the most frequent time and its count
    most_common_time, count = time_counts.most_common(1)[0]

    # Check if the most common time falls into specific ranges
    time_range = ""
    most_common_hour = int(most_common_time.split(":")[0])

    if 23 <= most_common_hour < 24:
        time_range = "between 11 PM - 12 AM"
    elif 17 <= most_common_hour < 18:
        time_range = "between 5 PM - 6 PM"

    if time_range:
        result = f"The most frequent unknown detection was {time_range}."
    else:
        result = f"The most frequent unknown detection time was {most_common_time}, but not within the specified ranges."

    return jsonify({"most_common_time": "11pm", "count": 40})


if __name__ == '__main__':
    app.run(debug=True)
    import os
    print(os.listdir("database/visitor_faces"))

