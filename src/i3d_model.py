import tensorflow as tf
from src.i3d_inception import Inception_Inflated3d

def build_i3d_model(
    input_shape=(16, 224, 224, 3),
    num_classes=825,
    dropout_rate=0.5,
    weights_path=None  # optional path to pretrained weights
):
    # Use the official I3D Inception model
    model = Inception_Inflated3d(
        include_top=True,
        weights=weights_path,
        input_shape=input_shape,
        classes=num_classes,
        dropout_prob=dropout_rate
    )
    return model

# Example usage:
# model = build_i3d_model(weights_path='i3d_kinetics_weights.h5')
# model.summary() 