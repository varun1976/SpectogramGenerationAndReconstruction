#!/usr/bin/env python3
import sys
import subprocess

def main():
    print("=" * 65)
    print("Spectrogram Generation & Reconstruction Platform Launcher")
    print("=" * 65)
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        print("Running unit tests...")
        cmd = [sys.executable, "-m", "pytest", "tests/"]
        subprocess.run(cmd)
    else:
        print("Launching Streamlit Web Application...")
        cmd = [sys.executable, "-m", "streamlit", "run", "app/main.py"]
        subprocess.run(cmd)

if __name__ == "__main__":
    main()
