# 🎯 OpenCV Face Recognition Attendance System

A fast, real-time attendance system powered by **OpenCV** and **face\_recognition**. Register faces through a sleek web UI, mark attendance via live webcam feed, and view records — all from your browser.

---

## ✨ Features

- **📸 Face Registration** — Upload photos to register new people into the system
- **🎥 Live Attendance** — Real-time face recognition via webcam with MJPEG streaming
- **📋 Dashboard** — View today's attendance at a glance with stats
- **📅 Records** — Browse all attendance records grouped by date
- **🗑️ Delete** — Remove registered people from the system
- **🌙 Dark Theme** — Modern, polished dark UI

---

## 🛠️ Tech Stack

| Layer      | Technology                        |
|------------|-----------------------------------|
| Backend    | Python, Flask                     |
| Vision     | OpenCV, face\_recognition, dlib   |
| Frontend   | HTML, CSS, Jinja2                 |
| Storage    | File-based (pickle + CSV)         |

---

## 📁 Project Structure

```
opencv/
├── app.py               # Flask web server & routes
├── attendance.py         # Attendance logic (mark, load, recognize)
├── encode_faces.py       # Face encoding utility
├── requirements.txt      # Python dependencies
├── known_faces/          # Registered face images (per-person folders)
├── static/               # Static assets (CSS)
└── templates/
    ├── index.html        # Dashboard
    ├── register.html     # Face registration page
    ├── mark.html         # Live attendance marking
    └── records.html      # Attendance records viewer
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Webcam (for live attendance marking)
- CMake (required by `dlib`)

### Installation

```bash
# Clone the repo
git clone https://github.com/devSoniak/opencv-facerecog-attendence.git
cd opencv-facerecog-attendence

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run

```bash
python app.py
```

Open **http://localhost:8080** in your browser.

---

## 📖 Usage

### 1. Register a Face
Navigate to **Register** → Enter name → Upload one or more photos → Submit.

### 2. Mark Attendance
Go to **Mark Attendance** → The webcam starts automatically → Recognized faces are marked in real-time.

### 3. View Records
Check **Records** to see attendance history grouped by date.

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

<p align="center">
  Built with ❤️ using OpenCV & Flask
</p>
