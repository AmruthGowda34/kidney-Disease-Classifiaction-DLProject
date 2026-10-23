
import tensorflow as tf
import numpy as np
from pathlib import Path

MODEL_PATH = Path("artifacts/training/model_v2.h5")
CLASS_NAMES = ["Cyst", "Normal", "Stone", "Tumor"]

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = tf.keras.models.load_model(
    str(MODEL_PATH),
    compile=False
)

def predict_kidney_image(image_path):
    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = tf.keras.utils.load_img(
        str(image_path),
        target_size=(224, 224),
        color_mode="rgb"
    )

    image_array = tf.keras.utils.img_to_array(image)

    # Assumes training used rescale=1./255.
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    probabilities = model.predict(image_array, verbose=0)[0]

    if len(probabilities) != len(CLASS_NAMES):
        raise ValueError("Model output does not match the four classes.")

    predicted_index = int(np.argmax(probabilities))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(probabilities[predicted_index]) * 100

    print(f"\nPredicted class: {predicted_class}")
    print(f"Predicted probability: {confidence:.2f}%")
    print("\nClass probabilities:")

    for name, probability in zip(CLASS_NAMES, probabilities):
        print(f"{name}: {probability * 100:.2f}%")

    return predicted_class, confidence


if __name__ == "__main__":
    image_path = input("Enter the kidney image path: ").strip().strip('"')
    predict_kidney_image(image_path)
