
import tensorflow as tf
import numpy as np
from pathlib import Path

# Load the trained AI model
model = tf.keras.models.load_model("waste_classifier.keras")

# Get the waste categories from the dataset folders
dataset_path = Path("dataset-resized")
class_names = sorted(
    folder.name for folder in dataset_path.iterdir()
    if folder.is_dir()
)

print("Smart Recycling Assistant")
print("Model loaded successfully!")
print("Categories:", class_names)

# Ask the user for an image path
image_path = input(
    "\nEnter the path of a waste image: "
).strip().strip('"')

# Check whether the image exists
if not Path(image_path).is_file():
    print("Error: Image file not found.")
else:
    # Load and prepare the image
    image = tf.keras.utils.load_img(
        image_path,
        target_size=(128, 128)
    )

    image_array = tf.keras.utils.img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)

    # Predict the waste category
    predictions = model.predict(image_array, verbose=0)
    predicted_index = int(np.argmax(predictions[0]))

    category = class_names[predicted_index]
    confidence = float(predictions[0][predicted_index]) * 100

    print("\nPrediction Result")
    print("-------------------------")
    print("Waste Category:", category)
    print(f"Model Confidence: {confidence:.2f}%")
