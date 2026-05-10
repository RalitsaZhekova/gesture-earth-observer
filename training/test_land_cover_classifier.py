from pathlib import Path
import json

import numpy as np
import tensorflow as tf
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "land_cover_classifier.keras"
CLASS_NAMES_PATH = PROJECT_ROOT / "models" / "class_names.json"

TEST_IMAGE_PATH = PROJECT_ROOT / "data" / "Forest" / "Forest_1.jpg"

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_NAMES_PATH, "r") as file:
    class_names = json.load(file)

image = Image.open(TEST_IMAGE_PATH).convert("RGB")
image = image.resize((224, 224))

image_array = np.array(image)
image_batch = np.expand_dims(image_array, axis=0)

predictions = model.predict(image_batch)

predicted_index = np.argmax(predictions[0])
predicted_class = class_names[predicted_index]
confidence = predictions[0][predicted_index]

print("Predicted class:", predicted_class)
print(f"Confidence: {confidence:.2%}")