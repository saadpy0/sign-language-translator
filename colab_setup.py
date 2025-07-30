#!/usr/bin/env python3
"""
Google Colab Setup Script for Video Sign Language Recognition
This script sets up the environment and mounts Google Drive for training.
"""

import os
import subprocess
import sys

def mount_google_drive():
    """Mount Google Drive in Colab."""
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("✅ Google Drive mounted successfully!")
        return True
    except ImportError:
        print("❌ This script is designed to run in Google Colab")
        return False

def install_dependencies():
    """Install required dependencies."""
    print("📦 Installing dependencies...")
    packages = [
        'tensorflow>=2.10.0',
        'opencv-python',
        'numpy',
        'matplotlib',
        'tqdm'
    ]
    
    for package in packages:
        print(f"   Installing {package}...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
    
    print("✅ Dependencies installed successfully!")

def create_directories():
    """Create necessary directories in Google Drive."""
    directories = [
        '/content/drive/MyDrive/asl_project',
        '/content/drive/MyDrive/asl_project/models',
        '/content/drive/MyDrive/asl_project/WLASL',
        '/content/drive/MyDrive/asl_project/WLASL/start_kit',
        '/content/drive/MyDrive/asl_project/WLASL/start_kit/frames',
        '/content/drive/MyDrive/asl_project/WLASL/start_kit/splits'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def main():
    """Main setup function."""
    print("🚀 Setting up Google Colab environment for Video Sign Language Recognition")
    print("=" * 70)
    
    # Mount Google Drive
    if not mount_google_drive():
        return
    
    # Install dependencies
    install_dependencies()
    
    # Create directories
    create_directories()
    
    print("\n" + "=" * 70)
    print("✅ Setup completed successfully!")
    print("\n📋 Next steps:")
    print("   1. Upload your 'wlasl_training_data.zip' to Google Drive")
    print("   2. Extract it to '/content/drive/MyDrive/asl_project/WLASL/'")
    print("   3. Clone your GitHub repository")
    print("   4. Run the training script")
    print("\n💡 Commands to run:")
    print("   # Extract the zip file")
    print("   !unzip /content/drive/MyDrive/wlasl_training_data.zip -d /content/drive/MyDrive/asl_project/")
    print("   # Clone your repo")
    print("   !git clone <your-repo-url>")
    print("   # Start training")
    print("   !python train_video_sign_recognition.py")

if __name__ == "__main__":
    main() 