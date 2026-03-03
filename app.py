"""
Flask Web Application for Attendance System
Routes: Dashboard, Register Face, Mark Attendance (live webcam), View Records
"""

import os
import pickle
import shutil
from datetime import date

import cv2
import numpy as np
import face_recognition
from flask import Flask, render_template, request, redirect, url_for, Response, flash, jsonify

from attendance import (
    load_encodings, recognize_and_mark, get_today_attendance,
    get_all_attendance, mark_attendance
)
from encode_faces import encode_known_faces, KNOWN_FACES_DIR, ENCODINGS_FILE

app = Flask(__name__)
app.secret_key = "attendance-system-secret-key-2026"

# ── Global state for video streaming ──
camera = None
streaming = False
recently_recognized = []  # names recognized in current session


def get_camera():
    """Get or create camera instance."""
    global camera
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return camera


def release_camera():
    """Release camera resource."""
    global camera, streaming
    streaming = False
    if camera is not None:
        camera.release()
        camera = None


def generate_frames():
    """Generator: yields MJPEG frames with face recognition overlay."""
    global streaming, recently_recognized
    streaming = True
    recently_recognized = []  # reset on new session

    data = load_encodings()

    cam = get_camera()
    if not cam.isOpened():
        streaming = False
        return

    frame_count = 0

    while streaming:
        ret, frame = cam.read()
        if not ret:
            break

        # Run face recognition only if encodings exist
        if data is not None and frame_count % 3 == 0:
            frame, recognized = recognize_and_mark(frame, data)
            if recognized:
                for name in recognized:
                    if name not in recently_recognized:
                        recently_recognized.append(name)
        elif data is None:
            # Show hint when no faces registered
            cv2.putText(frame, "Your face is not registered", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        frame_count += 1

        # Add timestamp overlay
        timestamp = f"{date.today().isoformat()}"
        cv2.putText(frame, timestamp, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    release_camera()


# ── Routes ──

@app.route("/")
def index():
    """Dashboard: show today's attendance."""
    today_records = get_today_attendance()
    all_records = get_all_attendance()

    # Count unique people registered
    registered_count = 0
    if os.path.exists(KNOWN_FACES_DIR):
        registered_count = len([
            d for d in os.listdir(KNOWN_FACES_DIR)
            if os.path.isdir(os.path.join(KNOWN_FACES_DIR, d))
        ])

    return render_template("index.html",
                           today_records=today_records,
                           total_registered=registered_count,
                           today_count=len(today_records),
                           today_date=date.today().strftime("%B %d, %Y"))


@app.route("/register", methods=["GET", "POST"])
def register():
    """Register a new face."""
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        photos = request.files.getlist("photos")

        if not name:
            flash("Please enter a name.", "error")
            return redirect(url_for("register"))

        if not photos or all(p.filename == '' for p in photos):
            flash("Please upload at least one photo.", "error")
            return redirect(url_for("register"))

        # Create directory for person
        person_dir = os.path.join(KNOWN_FACES_DIR, name)
        os.makedirs(person_dir, exist_ok=True)

        saved = 0
        for photo in photos:
            if photo.filename == '':
                continue
            ext = os.path.splitext(photo.filename)[1].lower()
            if ext not in ('.jpg', '.jpeg', '.png', '.bmp', '.webp'):
                continue
            filepath = os.path.join(person_dir, f"{name}_{saved + 1}{ext}")
            photo.save(filepath)
            saved += 1

        if saved > 0:
            # Re-encode all faces
            encode_known_faces()
            flash(f"Successfully registered '{name}' with {saved} photo(s)!", "success")
        else:
            flash("No valid photos were uploaded.", "error")

        return redirect(url_for("register"))

    # List registered people
    registered = []
    if os.path.exists(KNOWN_FACES_DIR):
        for name in sorted(os.listdir(KNOWN_FACES_DIR)):
            person_dir = os.path.join(KNOWN_FACES_DIR, name)
            if os.path.isdir(person_dir):
                count = len([f for f in os.listdir(person_dir)
                             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))])
                registered.append({"name": name, "photos": count})

    return render_template("register.html", registered=registered)


@app.route("/mark")
def mark():
    """Mark attendance page with live webcam feed."""
    return render_template("mark.html")


@app.route("/video_feed")
def video_feed():
    """MJPEG video stream endpoint."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/stop_feed")
def stop_feed():
    """Stop the video feed."""
    release_camera()
    return redirect(url_for("index"))


@app.route("/check_recognized")
def check_recognized():
    """API: returns list of recently recognized names for frontend polling."""
    global recently_recognized
    names = list(recently_recognized)
    return jsonify({"recognized": names})


@app.route("/records")
def records():
    """View all attendance records."""
    all_records = get_all_attendance()
    # Group by date
    dates = {}
    for r in all_records:
        d = r.get("date", "Unknown")
        if d not in dates:
            dates[d] = []
        dates[d].append(r)

    # Sort dates descending
    sorted_dates = sorted(dates.keys(), reverse=True)
    grouped = [(d, dates[d]) for d in sorted_dates]

    return render_template("records.html", grouped=grouped)


@app.route("/delete/<name>")
def delete_person(name):
    """Delete a registered person."""
    person_dir = os.path.join(KNOWN_FACES_DIR, name)
    if os.path.exists(person_dir):
        shutil.rmtree(person_dir)
        # Re-encode
        encode_known_faces()
        flash(f"Deleted '{name}' from the system.", "success")
    else:
        flash(f"Person '{name}' not found.", "error")
    return redirect(url_for("register"))


if __name__ == "__main__":
    os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
    print("╔══════════════════════════════════════════╗")
    print("║   OpenCV Attendance System               ║")
    print("║   http://localhost:8080                   ║")
    print("╚══════════════════════════════════════════╝")
    app.run(debug=True, host="0.0.0.0", port=8080)
