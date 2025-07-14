import pickle
import cv2
import face_recognition
import os
import tkinter as tk
from tkinter import simpledialog
import time
import pyttsx3 

DB_FOLDER="recordings/unknown"
ENCODINGS_FILE = "database/face_encodings.pkl"
UNKNOWN_FOLDER = "recordings/unknown"
RECORDINGS_FOLDER = "recordings"
VISITOR_FOLDER="database/visitor_faces"
KNOWN_FACES_FOLDER="database/known_faces"

# Ensure the recordings folder exists
os.makedirs(RECORDINGS_FOLDER, exist_ok=True)
# Keep track of already greeted names
greeted_names = set()

# Initialize the text-to-speech engine
engine = pyttsx3.init()
def speak(text):
    """Function to speak the given text"""
    engine.say(text)
    engine.runAndWait()
# Keep track of already greeted names
greeted_names = set()
import pickle
import os

ENCODINGS_FILE = "database/face_encodings.pkl"

def load_known_faces():
    """Function to load known faces and their encodings."""
    if not os.path.exists(ENCODINGS_FILE):
        return [], []  # Return empty lists if the encodings file does not exist

    with open(ENCODINGS_FILE, "rb") as f:
        known_faces, known_names = pickle.load(f)

    return known_faces, known_names

def recognize_faces(frame):
    global greeted_names

    with open(ENCODINGS_FILE, "rb") as f:
        known_faces, known_names = pickle.load(f)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    detected_names = []
    unknown_faces = []
    unknown_images = []

    for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.5)

        name = "Unknown"
        if True in matches:
            first_match_index = matches.index(True)
            name = known_names[first_match_index]
            if name not in greeted_names:
                print(f"HELLO {name} 👋")
                greeted_names.add(name)

        # Extract the folder path of the recognized face's owner
                owner_folder = os.path.join(KNOWN_FACES_FOLDER, name)  # Folder name corresponds to the owner's name
                owner_name = os.path.basename(owner_folder)  # Get the folder name (which is also the owner's name)

        # Dynamic greeting based on the owner's folder
                speak(f"Hello {name}, welcome Keerthi's home!")

        else:
            unknown_faces.append(face_encoding)
            unknown_images.append(frame[top:bottom, left:right])

        detected_names.append(name)
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    if unknown_faces:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = os.path.join(UNKNOWN_FOLDER, f"unknown_face_{timestamp}.jpg")
        cv2.imwrite(filename, frame)
        print(f"[INFO] Entire frame saved: {filename}")
    return frame, detected_names

import cv2

def start_recognition():
    cap = cv2.VideoCapture(0)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame, detected_names = recognize_faces(frame)
            cv2.imshow("Visitor Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break  # Exit on pressing 'q'

    except KeyboardInterrupt:
        print("\n[INFO] Keyboard Interrupt detected. Exiting gracefully...")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Camera released, windows closed.")



