import os
import pickle
import face_recognition

KNOWN_FACES_DIR = "database/known_faces/"
VISITOR_FACES_DIR = "database/visitor_faces/"
ENCODINGS_FILE = "database/face_encodings.pkl"

def encode_known_faces():
    known_faces = []
    known_names = []

    # Encode from known faces directory
    for name in os.listdir(KNOWN_FACES_DIR):
        person_path = os.path.join(KNOWN_FACES_DIR, name)
        if not os.path.isdir(person_path):
            continue

        for filename in os.listdir(person_path):
            image_path = os.path.join(person_path, filename)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)

            if encodings:
                known_faces.append(encodings[0])
                known_names.append(name)

    # Encode from visitor faces directory
    for name in os.listdir(VISITOR_FACES_DIR):
        person_path = os.path.join(VISITOR_FACES_DIR, name)
        if not os.path.isdir(person_path):
            continue

        for filename in os.listdir(person_path):
            image_path = os.path.join(person_path, filename)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)

            if encodings:
                known_faces.append(encodings[0])
                known_names.append(name)

    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump((known_faces, known_names), f)

    print("✅ Face encodings saved successfully!")

    # Load and display encoded names
    with open(ENCODINGS_FILE, "rb") as f:
        _, saved_names = pickle.load(f)
    
    print("📝 Encoded Faces:", saved_names)

if __name__ == "__main__":
    encode_known_faces()
