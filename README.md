# Eye Health Detection System 

An AI-based system that detects eye fatigue and analyzes screen usage using facial landmarks and machine learning.


##  Overview
This project monitors eye strain by analyzing blink rate, eye distance, and facial landmarks in real time. It helps users understand their screen usage patterns and prevent digital eye fatigue.



##  Features
- Eye fatigue detection
- Face landmark tracking
- Screen usage analysis
- Fatigue scoring system



##  Tech Stack
- Python
- Flask
- MediaPipe
- Machine Learning



##  Requirements
This project requires **Python 3.10** (MediaPipe compatibility).



##  Setup Instructions


### 1. Install Python 3.10
Check version:
```bash
python --version 
```
If not installed, download Python 3.10 from the official website.

### 2. Clone the repository
```bash
git clone https://github.com/riyantium/eye-health-detection-and-smart-screentime-analyzer.git
cd eye-health-detection-and-smart-screentime-analyzer
```

### 3. Create a virtual environment
```bash
python -m venv eye_env
```

Activate it (Windows):
```bash
eye_env\Scripts\activate
```

### 4. Install dependancies
```bash
pip install -r requirements.txt
```

### 5. Run the project
```bash
python api.py
```

## Project Structure

- api.py → backend server 
- fatigue.py → fatigue detection logic 
- ocular.py → eye analysis \n
- ml.py → machine learning model 
- index.html → frontend 

