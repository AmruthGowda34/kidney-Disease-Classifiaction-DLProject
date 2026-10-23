import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    def predict(self):
        # Load the trained model
        model_path = os.path.join(
            "artifacts", "training", "model_v2.h5"
        )
        model = load_model(model_path)

        # Load the image using the same target size as training
        test_image = image.load_img(
            self.filename,
            target_size=(224, 224)
        )

        # Convert image to an array
        test_image = image.img_to_array(test_image)

        # Apply the same preprocessing used during training
        test_image = tf.keras.applications.vgg16.preprocess_input(
            test_image
        )

        # Add batch dimension
        test_image = np.expand_dims(test_image, axis=0)

        # Predict
        result = model.predict(test_image, verbose=0)
        predicted_class = int(np.argmax(result, axis=1)[0])

        # Class order confirmed from your dataset
        class_names = ["Cyst", "Normal", "Stone", "Tumor"]
        prediction = class_names[predicted_class]

        # Highest model output score
        confidence = float(np.max(result)) * 100

        return [{
            "image": prediction,
            "confidence": round(confidence, 2)
        }]
