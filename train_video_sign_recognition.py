#!/usr/bin/env python3
"""
Video-based Sign Language Recognition Training Script
Trains a CNN+LSTM model on WLASL dataset for word-level sign recognition.
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf
from datetime import datetime
import matplotlib.pyplot as plt

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from video_data_loader import WLASLVideoDataLoader
from video_sign_model import VideoSignLanguageModel

def setup_gpu():
    """Configure GPU settings for optimal performance."""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Allow memory growth
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Found {len(gpus)} GPU(s). Memory growth enabled.")
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
    else:
        print("No GPU found. Using CPU.")

def create_model_directory():
    """Create models directory if it doesn't exist."""
    os.makedirs('models', exist_ok=True)

def save_training_config(config: dict, filename: str = 'training_config.json'):
    """Save training configuration to file."""
    with open(filename, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Training configuration saved to {filename}")

def plot_training_history(history, save_path: str = 'training_history.png'):
    """Plot and save training history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Training history plot saved to {save_path}")

def main():
    """Main training function."""
    print("🚀 Starting Video-based Sign Language Recognition Training")
    print("=" * 60)
    
    # Setup
    setup_gpu()
    create_model_directory()
    
    # Training configuration
    config = {
        'dataset': {
            'frames_dir': '/content/drive/MyDrive/asl/WLASL/start_kit/frames',
            'split_file': '/content/drive/MyDrive/asl/WLASL/start_kit/splits/WLASL100.json',
            'num_frames': 16,
            'frame_size': (224, 224),
            'batch_size': 16  # Reduced for memory efficiency
        },
        'model': {
            'lstm_units': 128,
            'dropout_rate': 0.5,
            'learning_rate': 1e-4
        },
        'training': {
            'epochs': 50,
            'patience': 15,
            'min_delta': 0.001
        }
    }
    
    # Save configuration
    save_training_config(config, '/content/drive/MyDrive/asl/training_config.json')
    
    print("\n📊 Loading dataset...")
    try:
        # Initialize data loader
        data_loader = WLASLVideoDataLoader(
            frames_dir=config['dataset']['frames_dir'],
            split_file=config['dataset']['split_file'],
            num_frames=config['dataset']['num_frames'],
            frame_size=config['dataset']['frame_size'],
            batch_size=config['dataset']['batch_size'],
            shuffle=True
        )
        
        print(f"✅ Dataset loaded successfully!")
        print(f"   - Number of classes: {data_loader.num_classes}")
        print(f"   - Train samples: {len(data_loader.train_samples)}")
        print(f"   - Val samples: {len(data_loader.val_samples)}")
        print(f"   - Test samples: {len(data_loader.test_samples)}")
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    print("\n🏗️ Building model...")
    try:
        # Initialize model
        model = VideoSignLanguageModel(
            num_classes=data_loader.num_classes,
            num_frames=config['dataset']['num_frames'],
            frame_height=config['dataset']['frame_size'][0],
            frame_width=config['dataset']['frame_size'][1],
            lstm_units=config['model']['lstm_units'],
            dropout_rate=config['model']['dropout_rate'],
            learning_rate=config['model']['learning_rate']
        )
        
        # Compile model
        model.compile_model()
        
        print("✅ Model built successfully!")
        model.summary()
        
    except Exception as e:
        print(f"❌ Error building model: {e}")
        return
    
    print("\n🧪 Testing data loading...")
    try:
        # Test data loading with a few samples
        test_frames, test_labels = data_loader.get_sample_data(num_samples=2)
        if len(test_frames) > 0:
            print(f"✅ Data loading test successful!")
            print(f"   - Test frames shape: {test_frames.shape}")
            print(f"   - Test labels shape: {test_labels.shape}")
            
            # Test model prediction
            test_predictions = model.predict(test_frames)
            print(f"   - Test predictions shape: {test_predictions.shape}")
        else:
            print("❌ No test data loaded")
            return
            
    except Exception as e:
        print(f"❌ Error testing data loading: {e}")
        return
    
    print("\n🎯 Starting training...")
    try:
        # Get data generators
        train_generator = data_loader.get_train_generator()
        val_generator = data_loader.get_val_generator()
        
        # Get training steps
        train_steps = data_loader.get_train_steps()
        val_steps = data_loader.get_val_steps()
        
        print(f"   - Training steps per epoch: {train_steps}")
        print(f"   - Validation steps per epoch: {val_steps}")
        
        # Get callbacks
        callbacks = model.get_callbacks(
            model_save_path='/content/drive/MyDrive/asl/models/video_sign_bilstm_best.h5',
            patience=config['training']['patience'],
            min_delta=config['training']['min_delta']
        )
        
        # Train the model
        print("\n🔥 Training started at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        history = model.train(
            train_generator=train_generator,
            val_generator=val_generator,
            train_steps=train_steps,
            val_steps=val_steps,
            epochs=config['training']['epochs'],
            callbacks=callbacks,
            verbose=1
        )
        
        print("\n✅ Training completed!")
        print(f"   - Final training accuracy: {history.history['accuracy'][-1]:.4f}")
        print(f"   - Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
        
        # Plot training history
        plot_training_history(history, '/content/drive/MyDrive/asl/training_history.png')
        
        # Save final model
        final_model_path = '/content/drive/MyDrive/asl/models/video_sign_bilstm_final.h5'
        model.save_model(final_model_path)
        
        print(f"\n🎉 Training completed successfully!")
        print(f"   - Best model saved to: /content/drive/MyDrive/asl/models/video_sign_bilstm_best.h5")
        print(f"   - Final model saved to: {final_model_path}")
        print(f"   - Training history plot saved to: /content/drive/MyDrive/asl/training_history.png")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 60)
    print("🎯 Next steps:")
    print("   1. Evaluate the model on test set")
    print("   2. Fine-tune the model if needed")
    print("   3. Create inference pipeline")
    print("   4. Build real-time prediction system")

if __name__ == "__main__":
    main() 