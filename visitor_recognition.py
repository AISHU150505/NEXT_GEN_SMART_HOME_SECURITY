import cv2
import numpy as np
import face_recognition
import os
import threading

VISITOR_DIR = "visitor_faces"
UNKNOWN_DIR = "unknown_faces"
os.makedirs(VISITOR_DIR, exist_ok=True)
os.makedirs(UNKNOWN_DIR, exist_ok=True)

visitor_encodings = {}

def load_known_faces():
    global visitor_encodings
    visitor_encodings = {}
    for filename in os.listdir(VISITOR_DIR):
        path = os.path.join(VISITOR_DIR, filename)
        image = face_recognition.load_image_file(path)
        encoding = face_recognition.face_encodings(image)
        if encoding:
            visitor_encodings[filename.split('.')[0]] = encoding[0]

def capture_image():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None

def recognize_visitor_thread():
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        rgb_frame = frame[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = []
        
        if face_locations:
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(list(visitor_encodings.values()), face_encoding)
            name = "Unknown"
            
            if True in matches:
                name = list(visitor_encodings.keys())[matches.index(True)]
            else:
                frame = capture_image()
                if frame is not None:
                    cv2.imshow("New Visitor - Enter Name", frame)
                    cv2.waitKey(1000)
                    name = input("Enter visitor name: ")
                    filename = f"{name}.jpg"
                    path = os.path.join(VISITOR_DIR, filename)
                    cv2.imwrite(path, frame)
                    load_known_faces()
            
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Visitor Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def start_recognition():
    threading.Thread(target=recognize_visitor_thread, daemon=True).start()

if __name__ == "__main__":
    load_known_faces()
    start_recognition()
