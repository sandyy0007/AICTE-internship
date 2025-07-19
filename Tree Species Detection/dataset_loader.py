# ----------------------------
# 🌳 dataset_loader.py
# By Abhay - Tree Species Detection
# ----------------------------

import tensorflow as tf
import matplotlib.pyplot as plt
import os

#  Dataset ka path (apna path daalna yahan)
dataset_path = r"C:\Users\chaya\Downloads\archive\Tree_Species_Dataset"

#  Training dataset load kar rahe hain (80% training, 20% validation setup hai)
train_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(224, 224),
    batch_size=32
)

#  Validation dataset bhi load kar liya
val_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(224, 224),
    batch_size=32
)

#  Classes dekh le ek baar confirm karne ke liye
class_names = train_ds.class_names
print("Classes mil gayi bhai:", class_names)

#  Kuch sample images dekh le for confirmation
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.show()




print(train_ds.class_names)
print("Number of classes:", len(train_ds.class_names))
