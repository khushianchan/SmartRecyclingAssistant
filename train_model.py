
import tensorflow as tf
from tensorflow.keras import layers, models

# Step 1: Set dataset folder path
DATASET_PATH = "dataset-resized"

# Step 2: Load training images (80%)
train_data = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(128, 128),
    batch_size=32
)

# Step 3: Load validation images (20%)
val_data = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(128, 128),
    batch_size=32
)

# Step 4: Display waste categories
class_names = train_data.class_names
print("Waste categories:", class_names)

# Step 5: Improve image loading speed
AUTOTUNE = tf.data.AUTOTUNE
train_data = train_data.prefetch(buffer_size=AUTOTUNE)
val_data = val_data.prefetch(buffer_size=AUTOTUNE)

# Step 6: Create CNN model
model = models.Sequential([
    layers.Input(shape=(128, 128, 3)),
    layers.Rescaling(1.0 / 255),
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(len(class_names), activation="softmax")
])

# Step 7: Compile the model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Step 8: Train the AI model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)

# Step 9: Save the trained model
model.save("waste_classifier.keras")

print("\nTraining completed successfully!")
print("Model saved as waste_classifier.keras")
