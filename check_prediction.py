
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model = load_model("artifacts/training/model_v2.h5")

image_path = input("Enter the full path of your test image: ").strip().strip('"')

img = image.load_img(image_path, target_size=(224, 224))
img_array = image.img_to_array(img)

img_array = tf.keras.applications.vgg16.preprocess_input(img_array)
img_array = np.expand_dims(img_array, axis=0)

scores = model.predict(img_array, verbose=0)[0]

class_names = ["Cyst", "Normal", "Stone", "Tumor"]

print("\nPrediction scores:")
for name, score in zip(class_names, scores):
    print(f"{name}: {score * 100:.4f}%")

print("\nPredicted class:", class_names[int(np.argmax(scores))])
