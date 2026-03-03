"""
Face Encoding Module
Scans known_faces/ directory structure and creates face encodings.
Directory structure: known_faces/<person_name>/<image_files>
"""

import os
import pickle
import face_recognition

KNOWN_FACES_DIR = "known_faces"
ENCODINGS_FILE = "encodings.pkl"


def encode_known_faces():
    """Encode all faces in the known_faces directory."""
    known_encodings = []
    known_names = []

    if not os.path.exists(KNOWN_FACES_DIR):
        os.makedirs(KNOWN_FACES_DIR)
        print(f"[INFO] Created '{KNOWN_FACES_DIR}/' directory. Add subfolders with person names containing their photos.")
        return

    people = [d for d in os.listdir(KNOWN_FACES_DIR) if os.path.isdir(os.path.join(KNOWN_FACES_DIR, d))]

    if not people:
        print("[WARN] No person directories found in known_faces/")
        # Remove stale encodings so unregistered faces show as Unknown
        if os.path.exists(ENCODINGS_FILE):
            os.remove(ENCODINGS_FILE)
            print(f"[INFO] Removed stale '{ENCODINGS_FILE}'")
        return

    for person_name in people:
        person_dir = os.path.join(KNOWN_FACES_DIR, person_name)
        image_files = [
            f for f in os.listdir(person_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))
        ]

        if not image_files:
            print(f"[WARN] No images found for '{person_name}'")
            continue

        print(f"[INFO] Encoding faces for: {person_name} ({len(image_files)} images)")

        for img_file in image_files:
            img_path = os.path.join(person_dir, img_file)
            try:
                image = face_recognition.load_image_file(img_path)
                encodings = face_recognition.face_encodings(image)

                if encodings:
                    known_encodings.append(encodings[0])
                    known_names.append(person_name)
                    print(f"  ✓ Encoded: {img_file}")
                else:
                    print(f"  ✗ No face found in: {img_file}")
            except Exception as e:
                print(f"  ✗ Error processing {img_file}: {e}")

    if known_encodings:
        data = {"encodings": known_encodings, "names": known_names}
        with open(ENCODINGS_FILE, "wb") as f:
            pickle.dump(data, f)
        print(f"\n[SUCCESS] Saved {len(known_encodings)} face encodings to '{ENCODINGS_FILE}'")
    else:
        # Remove stale encodings so unregistered faces show as Unknown
        if os.path.exists(ENCODINGS_FILE):
            os.remove(ENCODINGS_FILE)
            print(f"[INFO] Removed stale '{ENCODINGS_FILE}'")
        print("\n[WARN] No faces were encoded.")


if __name__ == "__main__":
    encode_known_faces()
