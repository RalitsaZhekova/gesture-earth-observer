from pathlib import Path
import json

import numpy as np
import tensorflow as tf
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "land_cover_classifier.keras"
DEFAULT_CLASS_NAMES_PATH = PROJECT_ROOT / "models" / "class_names.json"

IMAGE_SIZE = (224, 224)


class LandCoverClassifier:
    def __init__(
        self,
        model_path=DEFAULT_MODEL_PATH,
        class_names_path=DEFAULT_CLASS_NAMES_PATH,
    ):
        self.model_path = Path(model_path).resolve()
        self.class_names_path = Path(class_names_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        if not self.class_names_path.exists():
            raise FileNotFoundError(f"Class names file not found: {self.class_names_path}")

        self.model = tf.keras.models.load_model(self.model_path)

        with open(self.class_names_path, "r") as file:
            self.class_names = json.load(file)

    def predict_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image = image.resize(IMAGE_SIZE)

        image_array = np.array(image)
        image_batch = np.expand_dims(image_array, axis=0)

        predictions = self.model.predict(image_batch, verbose=0)

        predicted_index = int(np.argmax(predictions[0]))
        predicted_class = self.class_names[predicted_index]
        confidence = float(predictions[0][predicted_index])

        return {
            "class": predicted_class,
            "confidence": confidence,
            "probabilities":  {
                self.class_names[i]: float(predictions[0][i])
                for i in range(len(self.class_names))
            }
        }