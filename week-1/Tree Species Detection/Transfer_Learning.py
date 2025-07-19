import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

# -----------------------------------------
# Load dataset
# -----------------------------------------

train_ds = tf.keras.utils.image_dataset_from_directory(
    r'D:\Tree Species Detection\Tree_Dataset',           #  CHANGE THIS to your dataset folder
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(224, 224),
    batch_size=32
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    'path/to/data',           #  SAME dataset folder
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(224, 224),
    batch_size=32
)

# -----------------------------------------
# Check number of classes dynamically
# -----------------------------------------

num_classes = len(train_ds.class_names)
print("Classes found:", train_ds.class_names)
print("Number of classes:", num_classes)

# -----------------------------------------
# Build model with EfficientNetB0 base
# -----------------------------------------

base_model = EfficientNetB0(
    input_shape=(224,224,3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(num_classes, activation='softmax')
])

# -----------------------------------------
# Compile the model
# -----------------------------------------

model.compile(
    optimizer=Adam(),
    loss='sparse_categorical_crossentropy',   #  use 'sparse_' if labels are integers
    metrics=['accuracy']
)

# -----------------------------------------
# Train the model
# -----------------------------------------

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

# -----------------------------------------
# Save the model
# -----------------------------------------

model.save('efficientnetb0_model.h5')
print("✅ Model saved as efficientnetb0_model.h5")
