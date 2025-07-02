import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

def build_cnn_lstm_model(
    input_shape=(16, 64, 64, 3),  # (frames, height, width, channels)
    num_classes=825,
    cnn_filters=32,  # restored
    lstm_units=128,  # restored
    dropout_rate=0.2,  # reduced
    l2_reg=1e-6  # reduced
):
    l2 = regularizers.l2(l2_reg)
    inputs = layers.Input(shape=input_shape)
    # TimeDistributed CNN with L2 and Dropout
    x = layers.TimeDistributed(layers.Conv2D(cnn_filters, (3, 3), activation='relu', padding='same', kernel_regularizer=l2))(inputs)
    x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x)
    x = layers.TimeDistributed(layers.Dropout(dropout_rate))(x)
    x = layers.TimeDistributed(layers.Conv2D(cnn_filters * 2, (3, 3), activation='relu', padding='same', kernel_regularizer=l2))(x)
    x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x)
    x = layers.TimeDistributed(layers.Dropout(dropout_rate))(x)
    x = layers.TimeDistributed(layers.Flatten())(x)
    # LSTM for temporal modeling with Dropout and L2
    x = layers.Bidirectional(layers.LSTM(lstm_units, kernel_regularizer=l2, recurrent_regularizer=l2, dropout=dropout_rate, return_sequences=False))(x)
    # Classifier
    outputs = layers.Dense(num_classes, activation='softmax', kernel_regularizer=l2)(x)
    model = models.Model(inputs, outputs)
    return model

# Example usage:
# model = build_cnn_lstm_model()
# model.summary() 