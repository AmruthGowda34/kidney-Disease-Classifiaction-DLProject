import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

DATASET_PATH = (
    "artifacts/data_ingestion/"
    "CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/"
    "CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone"
)

MODEL_PATH = "artifacts/training/model_v2.h5"

datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.vgg16.preprocess_input,
    validation_split=0.20
)

validation_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(224, 224),
    batch_size=16,
    subset="validation",
    shuffle=False
)

model = load_model(MODEL_PATH)

probabilities = model.predict(validation_data, verbose=1)
predictions = probabilities.argmax(axis=1)
actual = validation_data.classes

print("\nClass mapping:")
print(validation_data.class_indices)

print("\nConfusion matrix:")
print(confusion_matrix(actual, predictions))

print("\nClassification report:")
print(classification_report(
    actual,
    predictions,
    target_names=list(validation_data.class_indices.keys()),
    zero_division=0
))
