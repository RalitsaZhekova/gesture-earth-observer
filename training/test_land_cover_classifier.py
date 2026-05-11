from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from land_cover_classifier import LandCoverClassifier


if len(sys.argv) > 1:
    image_path = Path(sys.argv[1])
else:
    image_path = PROJECT_ROOT / "data" / "Forest" / "Forest_1.jpg"

if not image_path.exists():
    raise FileNotFoundError(f"Image not found: {image_path}")

classifier = LandCoverClassifier()
result = classifier.predict_image(image_path)

print("Image:", image_path)
print("Predicted class:", result["class"])
print(f"Confidence: {result['confidence']:.2%}")