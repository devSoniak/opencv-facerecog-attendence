"""
Attendance Module
Handles face recognition from webcam and attendance logging to CSV.
"""

import os
import csv
import pickle
from datetime import datetime, date

import cv2
import numpy as np
import face_recognition

ENCODINGS_FILE = "encodings.pkl"
ATTENDANCE_FILE = "attendance.csv"
TOLERANCE = 0.5


def load_encodings():
    """Load known face encodings from pickle file."""
    if not os.path.exists(ENCODINGS_FILE):
        print("[ERROR] No encodings file found. Run encode_faces.py first.")
        return None
    with open(ENCODINGS_FILE, "rb") as f:
        data = pickle.load(f)
    # Guard against empty/corrupt encodings
    if not data or not data.get("encodings") or not data.get("names"):
        return None
    return data


def get_today_attendance():
    """Read today's attendance from CSV."""
    today = date.today().isoformat()
    records = []
    if not os.path.exists(ATTENDANCE_FILE):
        return records
    with open(ATTENDANCE_FILE, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("date") == today:
                records.append(row)
    return records


def get_all_attendance():
    """Read all attendance records from CSV."""
    records = []
    if not os.path.exists(ATTENDANCE_FILE):
        return records
    with open(ATTENDANCE_FILE, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
    return records


def is_already_marked(name):
    """Check if a person has already been marked present today."""
    today = date.today().isoformat()
    if not os.path.exists(ATTENDANCE_FILE):
        return False
    with open(ATTENDANCE_FILE, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("name") == name and row.get("date") == today:
                return True
    return False


def mark_attendance(name):
    """Log attendance for a person with current timestamp."""
    if is_already_marked(name):
        return False

    now = datetime.now()
    file_exists = os.path.exists(ATTENDANCE_FILE)

    with open(ATTENDANCE_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "date", "time", "status"])
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "name": name,
            "date": now.date().isoformat(),
            "time": now.strftime("%H:%M:%S"),
            "status": "Present"
        })
    return True


def recognize_and_mark(frame, data):
    """
    Recognize faces in a frame and mark attendance.
    Returns annotated frame and list of recognized names.
    """
    # Resize for faster processing
    small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small)
    face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

    recognized = []

    for encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(data["encodings"], encoding, tolerance=TOLERANCE)
        name = "Unknown"

        if True in matches:
            face_distances = face_recognition.face_distance(data["encodings"], encoding)
            best_match_idx = np.argmin(face_distances)
            if matches[best_match_idx]:
                name = data["names"][best_match_idx]
                was_new = mark_attendance(name)
                if was_new:
                    recognized.append(name)

        # Scale back up face locations
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw box and label
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        # Label background
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

    return frame, recognized


def run_webcam_attendance():
    """Standalone webcam attendance mode."""
    data = load_encodings()
    if data is None:
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam")
        return

    print("[INFO] Starting webcam attendance. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame, recognized = recognize_and_mark(frame, data)

        for name in recognized:
            print(f"  ✓ Attendance marked: {name}")

        cv2.imshow("Attendance System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Webcam closed.")


if __name__ == "__main__":
    run_webcam_attendance()
