
import tensorflow as tf

# Load pretrained VGG16
model = tf.keras.applications.VGG16(
    input_shape=(224, 224, 3),
    weights="imagenet",
    include_top=False
)

# Freeze earlier layers; unfreeze the final four
for layer in model.layers[:-4]:
    layer.trainable = False

for layer in model.layers[-4:]:
    layer.trainable = True

# Add the four-class output layer
x = tf.keras.layers.Flatten()(model.output)
output = tf.keras.layers.Dense(4, activation="softmax")(x)

full_model = tf.keras.Model(model.input, output)

full_model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("Total parameters:", full_model.count_params())
print(
    "Trainable parameters:",
    sum(
        tf.keras.backend.count_params(w)
        for w in full_model.trainable_weights
    )
)
print(
    "Trainable VGG16 layers:",
    [layer.name for layer in model.layers if layer.trainable]
)
