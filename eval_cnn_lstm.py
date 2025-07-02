import tensorflow as tf
from src.data_loader import load_wlasl_sequence_dataset
from src.cnn_lstm_model import build_cnn_lstm_model

MODEL_PATH = 'models/lstm/wlasl_cnn_lstm_best.keras'
FRAME_DATA_DIR = 'WLASL/start_kit/frame_data'
BATCH_SIZE = 32

# Load validation data only
def main():
    _, val_ds = load_wlasl_sequence_dataset(
        frame_data_dir=FRAME_DATA_DIR,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        seed=123
    )

    # Load the best model
    model = tf.keras.models.load_model(MODEL_PATH, compile=True)
    loss, acc = model.evaluate(val_ds)
    print(f'Validation accuracy: {acc:.4f}, loss: {loss:.4f}')

if __name__ == '__main__':
    main() 