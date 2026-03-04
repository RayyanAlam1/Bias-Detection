"""
Run this script to start the News Bias Classifier web app.
Usage:
    python run_app.py
Then open your browser at: http://127.0.0.1:5000
"""
import subprocess, sys, os

app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "webapp", "app.py")
subprocess.run([sys.executable, app_path])
