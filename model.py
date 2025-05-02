import ast
import cv2
import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
from sklearn.metrics import confusion_matrix

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from keras import models, layers, losses
import tensorflow as tf


df = pd.read_csv(r"df_proc.csv", encoding="utf-8")
df["target"] = df["target"].apply(ast.literal_eval)
class_names = ["Normal", "Diabetes", "Glaucoma", "Cataract", "Age related Macular Degeneration",
               "Hypertension", "Pathological Myopia", "Other diseases/abnormalities"]


def load_image(filename, target):
    img = cv2.imread(f"Training Images/{filename.decode('utf-8')}", cv2.IMREAD_COLOR)
    img = tf.cast(img, tf.float32) / 255.0
    return img, np.array(target, dtype=np.float32)


image_paths = df["filename"].values
image_paths_tensor = tf.constant(image_paths, dtype=tf.string)


targets = np.array(df["target"].tolist())
targets_tensor = tf.constant(targets, dtype=tf.float32)


dataset = tf.data.Dataset.from_tensor_slices(
    (image_paths_tensor, targets_tensor))


def load_data(x, y):
    img, label = tf.numpy_function(load_image, [x, y], [tf.float32, tf.float32])
    img.set_shape([128, 128, 3])
    label.set_shape([8])
    return img, label


size = 6393
dataset = dataset.map(load_data)
dataset = dataset.shuffle(size)

train_dataset = dataset.take(int(0.7 * len(image_paths_tensor)))
val_test_dataset = dataset.skip(int(0.7 * len(image_paths_tensor)))

val_dataset = val_test_dataset.take(int(0.15 * len(image_paths_tensor)))
test_dataset = val_test_dataset.skip(int(0.15 * len(image_paths_tensor)))

train_dataset = train_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(32).prefetch(tf.data.AUTOTUNE)


model = models.Sequential()
model.add(layers.Input(shape=(128, 128, 3)))
model.add(layers.Conv2D(filters=128, kernel_size=(
    3, 3), strides=1, activation='relu'))
model.add(layers.MaxPooling2D())

model.add(layers.Conv2D(filters=64, kernel_size=(
    3, 3), strides=1, activation='relu'))
model.add(layers.MaxPooling2D())

model.add(layers.Conv2D(filters=32, kernel_size=(
    3, 3), strides=1, activation='relu'))
model.add(layers.MaxPooling2D())

model.add(layers.Conv2D(filters=16, kernel_size=(
    3, 3), strides=1, activation='relu'))
model.add(layers.MaxPooling2D())

model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(8, activation='softmax'))


weights = compute_class_weight(class_weight="balanced", classes=np.unique(
    np.argmax(targets, axis=1)), y=np.argmax(targets, axis=1))
weights_tensor = tf.constant(weights, dtype=tf.float32)


def weights_compile(y_true, y_pred):
    weights = tf.reduce_sum(weights_tensor * y_true, axis=1)
    loss = losses.categorical_crossentropy(y_true, y_pred)
    return loss * weights


model.compile(optimizer='adam', loss=weights_compile, metrics=['accuracy'])


hist = model.fit(train_dataset, epochs=29, batch_size=32, validation_data=val_dataset)

model.save("ODM.keras")
# metrics = hist.history
# print(metrics)
# history_df = pd.DataFrame(metrics)
# history_df.to_csv('training_metrics.csv', index=False)


# plt.figure(figsize=(10, 6))
# plt.plot(metrics['loss'], label='Training Loss')
# plt.plot(metrics['val_loss'], label='Validation Loss')
# plt.title('Loss vs. Epochs')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.grid(True)
# plt.savefig('training_loss_vs_epochs.png')
# plt.close()


# plt.figure(figsize=(10, 6))
# plt.plot(metrics['accuracy'], label='Training Accuracy')
# plt.plot(metrics['val_accuracy'], label='Validation Accuracy')
# plt.title('Accuracy vs. Epochs')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.grid(True)
# plt.savefig('training_accuracy_vs_epochs.png')
# plt.close()


# y_true = []
# y_pred = []
# images = []

# for img, label in val_dataset:
#     pred = model.predict(img)

#     y_true.extend(np.argmax(label.numpy(), axis=1))
#     y_pred.extend(np.argmax(pred, axis=1))

#     images.extend(img.numpy())

# y_true = np.array(y_true)
# y_pred = np.array(y_pred)


# matrix = confusion_matrix(y_true, y_pred)
# plt.figure(figsize=(8, 6))
# sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues',
#             xticklabels=class_names, yticklabels=class_names)
# plt.xlabel('Prediction')
# plt.ylabel('Truth')
# plt.savefig('confusion_matrix.png')
# plt.close() 


# missed = np.where(y_true != y_pred)[0]

# plt.figure(figsize=(12, 8))
# for i, idx in enumerate(missed[:6]):
#     plt.subplot(2, 3, i + 1)
#     plt.imshow(images[idx])
#     plt.title(f"True: {class_names[y_true[idx]]}\nPred: {class_names[y_pred[idx]]}")
#     plt.axis('off')
# plt.suptitle("Misses")
# plt.tight_layout()
# plt.savefig('misses.png') 
# plt.close()