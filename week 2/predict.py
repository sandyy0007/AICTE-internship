import sys
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# Load the model
model_path = 'tree_transfer_mobilenetv2.h5'
model = tf.keras.models.load_model(model_path)
print(f"✅ Loaded model from {model_path}")

# Set image size
img_height, img_width = 224, 224

# --------- 🔍 Dynamically Load Class Names ---------
# Point to the original training dataset location
dataset_path = r"D:\Final_tree_species_project\Tree_Species_Dataset"  # CHANGE this to your real dataset folder

if not os.path.exists(dataset_path):
    print("❌ Dataset path not found:", dataset_path)
    sys.exit()

# Load once just to get class names
temp_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=1
)
class_names = temp_ds.class_names
print("📚 Class Names:", class_names)

# --------- 🔍 Image Path Check ---------
if len(sys.argv) < 2:
    print("❌ Please provide the path to an image.")
    sys.exit()

img_path = sys.argv[1]
if not os.path.exists(img_path):
    print(f"❌ Image not found at: {img_path}")
    sys.exit()

# Load and preprocess the image
img = image.load_img(img_path, target_size=(img_height, img_width))
img_array = image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create batch axis

# Predict
predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])
predicted_class = class_names[np.argmax(score)]

print(f"🌳 Predicted Tree Species: {predicted_class} ({100 * np.max(score):.2f}% confidence)")
