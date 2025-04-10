import ast,cv2
import numpy as np
import pandas as pd  
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from keras import models, layers, callbacks, utils
import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


df=pd.read_csv(r"df_proc.csv", encoding="utf-8")
df["target"] = df["target"].apply(ast.literal_eval)


def load_image(filename, target):
    img = cv2.imread(f"Training Images/{filename.decode('utf-8')}", cv2.IMREAD_COLOR)
    img = tf.cast(img, tf.float32) / 255.0  
    return img, np.array(target, dtype=np.float32)

image_paths = df["filename"].values
image_paths_tensor = tf.constant(image_paths, dtype=tf.string)

targets = np.array(df["target"].tolist())
targets_tensor = tf.constant(targets, dtype=tf.float32)


training_dataset = tf.data.Dataset.from_tensor_slices((image_paths_tensor, targets_tensor))

    
def load_data(x, y):
    img, label = tf.numpy_function(load_image, [x, y], [tf.float32, tf.float32])
    img.set_shape([512, 512, 3])
    label.set_shape([8])
    return img, label

training_dataset = training_dataset.map(load_data)
training_dataset = training_dataset.shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)

model = models.Sequential()
model.add(layers.Input(shape=(512, 512, 3)))
model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), strides=1, activation='relu'))
model.add(layers.MaxPooling2D())
model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), strides=1, activation='relu'))
model.add(layers.MaxPooling2D())
model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), strides=1, activation='relu'))
model.add(layers.MaxPooling2D())
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(8, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# print(model.summary())
for inputs, labels in training_dataset.take(1):
    print("Input shape:", inputs.shape)
    print("Label shape:", labels)
print(model.input_shape)
callback = callbacks.TensorBoard(log_dir="Training Images")

hist = model.fit(training_dataset, epochs=20, batch_size=32, callbacks=callback)
