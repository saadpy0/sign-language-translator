import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import numpy as np

class VideoSignLanguageModel:
    """
    Video-based Sign Language Recognition Model
    Uses CNN+LSTM architecture for temporal modeling of sign language videos.
    """
    
    def __init__(self, 
                 num_classes: int,
                 num_frames: int = 16,
                 frame_height: int = 224,
                 frame_width: int = 224,
                 num_channels: int = 3,
                 lstm_units: int = 128,
                 dropout_rate: float = 0.5,
                 learning_rate: float = 1e-4):
        """
        Initialize the video sign language model.
        
        Args:
            num_classes: Number of sign language classes (glosses)
            num_frames: Number of frames per video
            frame_height: Height of input frames
            frame_width: Width of input frames
            num_channels: Number of color channels
            lstm_units: Number of LSTM units
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimization
        """
        self.num_classes = num_classes
        self.num_frames = num_frames
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.num_channels = num_channels
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        
        self.model = self._build_model()
    
    def _build_model(self) -> Model:
        """Build the CNN+LSTM model architecture."""
        
        # Input layer for video frames
        video_input = layers.Input(
            shape=(self.num_frames, self.frame_height, self.frame_width, self.num_channels),
            name='video_input'
        )
        
        # CNN backbone for frame feature extraction
        # Use MobileNetV2 as the base CNN (pre-trained on ImageNet)
        cnn_base = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(self.frame_height, self.frame_width, self.num_channels)
        )
        
        # Freeze the CNN backbone initially for faster training
        cnn_base.trainable = False
        
        # Apply CNN to each frame using TimeDistributed
        frame_features = layers.TimeDistributed(
            cnn_base,
            name='frame_encoder'
        )(video_input)
        
        # Global average pooling to reduce spatial dimensions
        frame_features = layers.TimeDistributed(
            layers.GlobalAveragePooling2D(),
            name='frame_pooling'
        )(frame_features)
        
        # Now we have shape: (batch_size, num_frames, 1280)
        
        # Bidirectional LSTM for temporal modeling
        lstm_output = layers.Bidirectional(
            layers.LSTM(self.lstm_units, return_sequences=True),
            name='bilstm_1'
        )(frame_features)
        
        # Second LSTM layer
        lstm_output = layers.Bidirectional(
            layers.LSTM(self.lstm_units // 2, return_sequences=False),
            name='bilstm_2'
        )(lstm_output)
        
        # Dropout for regularization
        lstm_output = layers.Dropout(self.dropout_rate, name='dropout')(lstm_output)
        
        # Dense layers for classification
        dense_output = layers.Dense(512, activation='relu', name='dense_1')(lstm_output)
        dense_output = layers.Dropout(self.dropout_rate, name='dropout_2')(dense_output)
        
        dense_output = layers.Dense(256, activation='relu', name='dense_2')(dense_output)
        dense_output = layers.Dropout(self.dropout_rate, name='dropout_3')(dense_output)
        
        # Output layer
        output = layers.Dense(self.num_classes, activation='softmax', name='classifier')(dense_output)
        
        # Create model
        model = Model(inputs=video_input, outputs=output, name='VideoSign_BiLSTM_MobileNetV2')
        
        return model
    
    def compile_model(self):
        """Compile the model with appropriate loss and optimizer."""
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def unfreeze_cnn_backbone(self, unfreeze_layers: int = 10):
        """
        Unfreeze the last few layers of the CNN backbone for fine-tuning.
        
        Args:
            unfreeze_layers: Number of layers to unfreeze from the end
        """
        # Get the CNN base model
        cnn_base = self.model.get_layer('frame_encoder').layer
        
        # Unfreeze the last few layers
        for layer in cnn_base.layers[-unfreeze_layers:]:
            layer.trainable = True
        
        # Recompile with a lower learning rate for fine-tuning
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate * 0.1),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"Unfroze last {unfreeze_layers} layers of CNN backbone")
    
    def get_callbacks(self, 
                     model_save_path: str = 'models/video_sign_bilstm_best.h5',
                     patience: int = 10,
                     min_delta: float = 0.001) -> list:
        """
        Get training callbacks.
        
        Args:
            model_save_path: Path to save the best model
            patience: Patience for early stopping
            min_delta: Minimum improvement for early stopping
            
        Returns:
            List of callbacks
        """
        callbacks = [
            # Model checkpoint to save best model
            ModelCheckpoint(
                filepath=model_save_path,
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                mode='max',
                verbose=1
            ),
            
            # Early stopping to prevent overfitting
            EarlyStopping(
                monitor='val_accuracy',
                patience=patience,
                min_delta=min_delta,
                mode='max',
                verbose=1,
                restore_best_weights=True
            ),
            
            # Reduce learning rate when plateau is reached
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        return callbacks
    
    def train(self, 
              train_generator,
              val_generator,
              train_steps: int,
              val_steps: int,
              epochs: int = 50,
              callbacks: list = None,
              verbose: int = 1):
        """
        Train the model.
        
        Args:
            train_generator: Training data generator
            val_generator: Validation data generator
            train_steps: Number of training steps per epoch
            val_steps: Number of validation steps per epoch
            epochs: Number of training epochs
            callbacks: List of callbacks
            verbose: Verbosity level
        """
        if callbacks is None:
            callbacks = self.get_callbacks()
        
        # Train the model
        history = self.model.fit(
            train_generator,
            steps_per_epoch=train_steps,
            validation_data=val_generator,
            validation_steps=val_steps,
            epochs=epochs,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return history
    
    def predict(self, video_frames: np.ndarray) -> np.ndarray:
        """
        Predict sign language class for a video.
        
        Args:
            video_frames: Video frames of shape (num_frames, height, width, channels)
            
        Returns:
            Predicted class probabilities
        """
        # Add batch dimension if needed
        if len(video_frames.shape) == 4:
            video_frames = np.expand_dims(video_frames, axis=0)
        
        predictions = self.model.predict(video_frames)
        return predictions
    
    def predict_class(self, video_frames: np.ndarray) -> int:
        """
        Predict the most likely sign language class.
        
        Args:
            video_frames: Video frames
            
        Returns:
            Predicted class index
        """
        predictions = self.predict(video_frames)
        return np.argmax(predictions, axis=1)[0]
    
    def summary(self):
        """Print model summary."""
        self.model.summary()
    
    def save_model(self, filepath: str):
        """Save the model to file."""
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a saved model."""
        self.model = tf.keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")

# Example usage
if __name__ == "__main__":
    # Test model creation
    model = VideoSignLanguageModel(
        num_classes=100,  # WLASL100 has 100 classes
        num_frames=16,
        frame_height=224,
        frame_width=224
    )
    
    model.compile_model()
    model.summary()
    
    # Test with dummy data
    dummy_input = np.random.random((2, 16, 224, 224, 3))
    predictions = model.predict(dummy_input)
    print(f"Predictions shape: {predictions.shape}") 