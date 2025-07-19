
# TREE SPECIES DETECTION PROJECT


import os
import tensorflow as tf
import matplotlib.pyplot as plt

#  Ye warning avoid karne ke liye
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

#  Dataset ka path (yahan apna path daalna hamesha)
dataset_path = r"C:\Users\chaya\Downloads\archive\Tree_Species_Dataset"

#  Dataset load kar rahe hain - 80% training, 20% validation
train_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(224, 224),
    batch_size=32
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(224, 224),
    batch_size=32
)

#  Classes check karlo ek baar
class_names = train_ds.class_names
print("Classes mil gayi bhai:", class_names)

#  Kuch sample images dekh lete hain for confirmation
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.show()

#  Dataset ko fast load karne ke liye optimize kar diya
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

#  Apna CNN model bana rahe hain ab
model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(224, 224, 3)),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(class_names), activation='softmax')
])

#  Model compile kar diya
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#  Model training start - thoda time lagega
epochs = 10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

#  Final evaluation
test_loss, test_acc = model.evaluate(val_ds)
print(f"\nTest Accuracy: {test_acc:.3f}")

#  Model save kar diya for future use
model.save("tree_species_model.h5")
print("Model saved as tree_species_model.h5 bhai")

#  Training graph dekh le for understanding
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(epochs)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Accuracy ka Graph')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Loss ka Graph')
plt.show()
