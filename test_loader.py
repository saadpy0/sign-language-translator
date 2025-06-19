from src.data_loader import load_asl_dataset
import matplotlib.pyplot as plt
import string

label_map = dict(enumerate(string.ascii_uppercase))


train_ds, val_ds = load_asl_dataset("data/raw/archive/asl_alphabet_train/asl_alphabet_train", img_size=(64, 64), batch_size=16)

for images, labels in train_ds.take(1):
    plt.figure(figsize=(10, 5))
    for i in range(8):
        ax = plt.subplot(2, 4, i + 1)
        plt.imshow(images[i].numpy())
        plt.title(f"Label: {labels[i].numpy()}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

