import tensorflow as tf
from tensorflow.keras import layers, models


def build_video_sign_bilstm_model(
    input_shape=(16, 224, 224, 3),
    num_classes=825,
    lstm_units=128,
    cnn_trainable=False,
    dropout_rate=0.5
):
    """
    Builds a video-based sign language recognition model using MobileNetV2 as frame encoder
    and a Bidirectional LSTM for temporal modeling.

    Args:
        input_shape (tuple): Shape of input (frames, height, width, channels)
        num_classes (int): Number of output classes (glosses)
        lstm_units (int): Number of units in the LSTM layer
        cnn_trainable (bool): Whether to fine-tune the CNN backbone
        dropout_rate (float): Dropout rate after LSTM

    Returns:
        tf.keras.Model: Compiled Keras model
    """
    frames, height, width, channels = input_shape
    # Frame encoder: MobileNetV2
    cnn_base = tf.keras.applications.MobileNetV2(
        input_shape=(height, width, channels),
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    cnn_base.trainable = cnn_trainable

    # Input: (batch, frames, height, width, channels)
    video_input = layers.Input(shape=input_shape, name="video_input")
    # TimeDistributed applies the CNN to each frame
    x = layers.TimeDistributed(cnn_base, name="frame_encoder")(video_input)
    # x shape: (batch, frames, cnn_features)
    x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=False), name="bilstm")(x)
    x = layers.Dropout(dropout_rate, name="dropout")(x)
    output = layers.Dense(num_classes, activation='softmax', name="classifier")(x)

    model = models.Model(inputs=video_input, outputs=output, name="VideoSign_BiLSTM_MobileNetV2")
    return model 