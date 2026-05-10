from pathlib import Path
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

if not DATA_DIR.exists():
    raise FileNotFoundError(f"Could not find dataset folder: {DATA_DIR}")

dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    image_size=(224, 224),
    batch_size=32,
    shuffle=True,
)

print("\nClasses found:")
print(dataset.class_names)

print("\nNumber of classes:")
print(len(dataset.class_names))

for images, labels in dataset.take(1):
    print("\nImage batch shape:", images.shape)
    print("Label batch shape:", labels.shape)
    print("Example labels:", labels[:10].numpy())