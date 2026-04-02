import os
import subprocess
import sys
import shutil

def build_exe():
    print("--- Starting SentinelAI Build Process ---")
    
    # 1. Install PyInstaller if missing
    try:
        import PyInstaller
    except ImportError:
        print("Installing PyInstaller...")
        # Ensure we use the current interpreter's pip
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
        except subprocess.CalledProcessError as e:
            print(f"\nERROR: Could not install PyInstaller. {e}")
            print("Check if your disk is full (e.g., D: drive).")
            print("Try creating a virtual environment on the C: drive (e.g., 'python -m venv venv') and activating it.")
            sys.exit(1)

    # 2. Create .env.example if it doesn't exist
    if not os.path.exists(".env.example"):
        with open(".env.example", "w") as f:
            f.write("PVPORCUPINE_PRIVATE_KEY=your_key_here\n")
            f.write("GEMINI_API_KEY=your_key_here\n")
            f.write("OPEN_AI_API_KEY=your_key_here\n")
            f.write("HOME_ASSISTANT_URL=\n")
            f.write("HOME_ASSISTANT_TOKEN=\n")

    # 3. Run PyInstaller
    print("Running PyInstaller...")
    subprocess.check_call(["pyinstaller", "--noconfirm", "sentinel.spec"])

    print("\n--- Build Complete! ---")
    print("Your shareable app is in the 'dist/SentinelAI' folder.")
    print("Note: Users will need to install Playwright browsers and Ollama manually or via your installer.")

if __name__ == "__main__":
    build_exe()
