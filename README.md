# 👁️ Eye Health Detection & Smart Screen-Time Analyzer

An AI-powered computer-vision system that monitors **eye fatigue and screen-related visual strain in real time** using facial landmarks, eye-movement analysis, and machine learning.

The system analyzes webcam input to estimate indicators such as **blink frequency, eye openness, eye distance, and prolonged visual focus**, then combines them into a fatigue score that helps users recognize potentially unhealthy screen-use patterns.

---

## 🚀 Why This Project?

Long periods of screen exposure can be associated with symptoms such as tired eyes, reduced blinking, and visual discomfort.

Instead of relying only on a timer, this project attempts to detect **observable behavioral indicators of eye fatigue** directly from a webcam.

### The system aims to answer:

> **"Is the user showing signs of increasing eye fatigue while using a screen?"**

---

## ✨ Key Features

### 👁️ Real-Time Eye Analysis

- Detects facial and eye landmarks using MediaPipe
- Tracks eye openness and eye geometry
- Monitors blink activity
- Calculates eye-related features continuously

### 🧠 Fatigue Detection

- Extracts relevant features from facial landmarks
- Processes them through fatigue-detection logic / ML
- Generates a real-time fatigue score
- Categorizes the user's current fatigue state

### 💻 Screen-Usage Analysis

- Tracks continuous screen sessions
- Monitors prolonged periods of visual focus
- Combines behavioral indicators with session duration
- Can be extended to provide break recommendations

### 📊 Fatigue Scoring

The system converts multiple eye-related indicators into an interpretable fatigue score.

Example:

```text
Fatigue Score: 72 / 100

Status: HIGH FATIGUE ⚠️

Indicators:
✓ Reduced blink activity
✓ Increased eye closure duration
✓ Prolonged screen session
````

---

# 🧠 How It Works

The overall pipeline is:

```text
Webcam
   ↓
Face Detection
   ↓
Facial Landmark Extraction
   ↓
Eye Landmark Detection
   ↓
Feature Extraction
   ↓
Fatigue Analysis / ML Model
   ↓
Fatigue Score
   ↓
Flask API
   ↓
Web Dashboard
```

### Feature Extraction

The system can extract features such as:

* Blink frequency
* Eye Aspect Ratio (EAR)
* Eye openness
* Eye closure duration
* Inter-eye distance
* Facial landmark positions
* Continuous screen-session duration

These features are then used to estimate the user's fatigue state.

---

# 🛠️ Tech Stack

| Technology              | Purpose                           |
| ----------------------- | --------------------------------- |
| **Python**              | Core application logic            |
| **MediaPipe**           | Facial and eye landmark detection |
| **OpenCV**              | Webcam/video processing           |
| **Machine Learning**    | Fatigue classification/scoring    |
| **Flask**               | Backend API                       |
| **HTML/CSS/JavaScript** | Frontend dashboard                |

---

# 📂 Project Structure

```text
eye-health-detection-and-smart-screentime-analyzer/
│
├── api.py
│   └── Flask backend / API server
│
├── fatigue.py
│   └── Fatigue detection and scoring logic
│
├── ocular.py
│   └── Eye analysis and ocular feature extraction
│
├── ml.py
│   └── Machine-learning model / prediction logic
│
├── index.html
│   └── Frontend dashboard
│
├── requirements.txt
│   └── Python dependencies
│
└── README.md
```

---

# ⚙️ Installation

## 1. Install Python

This project currently requires **Python 3.10** because of MediaPipe compatibility.

Check your Python version:

```bash
python --version
```

If Python 3.10 is not installed, download it from the official Python website.

---

## 2. Clone the Repository

```bash
git clone https://github.com/riyantium/eye-health-detection-and-smart-screentime-analyzer.git

cd eye-health-detection-and-smart-screentime-analyzer
```

---

## 3. Create a Virtual Environment

```bash
python -m venv eye_env
```

### Windows

```bash
eye_env\Scripts\activate
```

### macOS / Linux

```bash
source eye_env/bin/activate
```

---

## 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

Start the Flask backend:

```bash
python api.py
```

Then open the local application in your browser using the address displayed by Flask.

Allow the browser/application to access your webcam when prompted.

---

# 📊 Example System Output

```text
--------------------------------
      EYE HEALTH MONITOR
--------------------------------

Blink Rate       : 14 blinks/min
Eye Openness     : Normal
Screen Session   : 48 minutes
Fatigue Score    : 64 / 100

Status           : MODERATE FATIGUE

Recommendation:
Consider taking a short screen break.
--------------------------------
```

---

# 🧩 Core Components

## 1. Facial Landmark Detection

MediaPipe detects facial landmarks from the webcam feed.

These landmarks provide the geometric information required to analyze the eyes and surrounding facial regions.

---

## 2. Ocular Analysis

`ocular.py` processes eye landmarks and extracts useful measurements such as:

```text
Eye openness
Blink events
Eye Aspect Ratio
Eye distance
Eye closure duration
```

---

## 3. Fatigue Detection

`fatigue.py` combines the extracted signals to estimate the current fatigue level.

The system can classify the user into states such as:

```text
LOW
MODERATE
HIGH
```

---

## 4. Machine Learning

`ml.py` contains the machine-learning component used to process extracted features and generate fatigue predictions.

The architecture can be extended with a trained model using features collected from multiple users and sessions.

---

# 🔬 AI Pipeline

The important part of the project is not simply detecting a face.

The system follows:

```text
Raw Webcam Frames
        ↓
Facial Landmarks
        ↓
Eye Landmarks
        ↓
Numerical Features
        ↓
Temporal / Behavioral Analysis
        ↓
Machine Learning
        ↓
Fatigue Prediction
        ↓
User Feedback
```

This makes the project a **computer-vision + machine-learning pipeline**, rather than simply a webcam application.

---

# 🔐 Privacy

The system is designed around webcam-based analysis.

A production version should follow privacy-first principles:

* Process webcam frames locally whenever possible
* Avoid storing raw video
* Store only required numerical features
* Clearly inform users when the camera is active
* Provide an option to stop monitoring

---

# 🔮 Future Improvements

Potential extensions include:

### 🤖 Better AI

* Personalized fatigue models
* Temporal deep-learning models
* User-specific baseline calibration
* Improved fatigue classification

### 📈 Analytics

* Daily/weekly fatigue reports
* Screen-session history
* Blink-rate trends
* Fatigue progression graphs

### 🔔 Intelligent Recommendations

* Automatic break reminders
* Adaptive break intervals
* 20-20-20 rule reminders
* Fatigue-based notifications

### 🖥️ Smart Screen-Time Monitoring

* Application-specific usage tracking
* Continuous-session detection
* Productivity vs. fatigue analysis

### 🔒 Privacy

* Fully local inference
* No raw webcam storage
* Local encrypted analytics

---

# 🎯 Project Goal

The long-term goal is to evolve the system from a simple **eye-fatigue detector** into an intelligent **digital eye-wellness assistant** capable of understanding a user's screen-use behavior and providing personalized interventions.

```text
Detect → Analyze → Predict → Recommend
```

---

# ⭐ If You Find This Project Useful

Consider giving the repository a ⭐ and contributing ideas or improvements.

```
```
