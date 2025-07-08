from video_sign_bilstm_model import build_video_sign_bilstm_model

if __name__ == "__main__":
    # Example parameters (adjust as needed)
    input_shape = (16, 224, 224, 3)
    num_classes = 825
    model = build_video_sign_bilstm_model(input_shape=input_shape, num_classes=num_classes)
    model.summary() 