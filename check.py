import pickle

ENCODINGS_FILE = "database/face_encodings.pkl"

def display_all_names():
    try:
        with open(ENCODINGS_FILE, "rb") as f:
            _, known_names = pickle.load(f)

        if known_names:
            print("✅ Stored Names in Encodings File:")
            for name in set(known_names):  # Use set() to remove duplicates
                print(name)
        else:
            print("❌ No names found in the encodings file.")
    except FileNotFoundError:
        print("❌ Encodings file not found. Run encode_known_faces() first.")

if __name__ == "__main__":
    display_all_names()
