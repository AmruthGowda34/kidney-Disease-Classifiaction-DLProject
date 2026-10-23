import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time
import math
from pathlib import Path
from cnnClassifier.entity.config_entity import TrainingConfig
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    
    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )

    def train_valid_generator(self):

        datagenerator_kwargs = dict(
            preprocessing_function=tf.keras.applications.vgg16.preprocess_input,
            validation_split=0.20
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )
        
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)
    
    
    def train(self):
        self.steps_per_epoch = math.ceil(
        self.train_generator.samples /
        self.train_generator.batch_size
    )

        self.validation_steps = math.ceil(
        self.valid_generator.samples /
        self.valid_generator.batch_size
    )

        # Calculate weights for each class
        class_weights_array = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(self.train_generator.classes),
        y=self.train_generator.classes
    )

        class_weights = {
        int(class_id): float(weight)
        for class_id, weight in enumerate(class_weights_array)
    }

        print("Class mapping:", self.train_generator.class_indices)
        print("Class weights:", class_weights)

        checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(self.config.trained_model_path),
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )

        self.model.fit(
        self.train_generator,
        epochs=self.config.params_epochs,
        steps_per_epoch=self.steps_per_epoch,
        validation_data=self.valid_generator,
        validation_steps=self.validation_steps,
        callbacks=[checkpoint],
        class_weight=class_weights
    )
