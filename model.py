from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
import pickle 
import numpy as np 
import sys


batch_size = 512
learning_rate = 0.01
epochs = 4


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=(1024*4))])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    print(e)

# preparing dataset 
with open('data.pickle', 'rb') as file: 
    data = pickle.load(file)
images, labels = map(list, zip(*data))
for i in range(len(labels)): 
    labels[i] = int(labels[i])
labels = np.array(to_categorical(labels, num_classes=14))
images = np.array(images).reshape(-1, 45, 45, 1) / 255

dataset = tf.data.Dataset.from_tensor_slices((images, labels))\
    .shuffle(buffer_size=len(images))\
    .batch(batch_size=batch_size)

model = Sequential([ 
    Conv2D(32, (3, 3), input_shape=(45, 45, 1), activation='elu', padding='same'),
    Conv2D(32, (3, 3), activation='elu', padding='same'),
    MaxPool2D((2, 2), padding='same'),

    Conv2D(64, (3, 3), activation='elu', padding='same'),
    Conv2D(64, (3, 3), activation='elu', padding='same'),
    Conv2D(64, (3, 3), activation='elu', padding='same'),
    MaxPool2D((2, 2), padding='same'),

    Conv2D(128, (3, 3), activation='elu', padding='same'),
    Conv2D(128, (3, 3), activation='elu', padding='same'), 
    Conv2D(128, (3, 3), activation='elu', padding='same'),  
    MaxPool2D((2, 2), padding='same'), 

    Conv2D(256, (3, 3), activation='elu', padding='same'),
    Conv2D(256, (3, 3), activation='elu', padding='same'),
    Conv2D(256, (3, 3), activation='elu', padding='same'),
    MaxPool2D((2, 2), padding='same'), 
    Flatten(),

    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(14, activation='softmax')
])

crossentropy = tf.keras.losses.categorical_crossentropy

@tf.function
def apply_gradients(images, labels):
    with tf.GradientTape() as tape:
        prediction = model(images, training=True)
        loss = crossentropy(y_true=labels, y_pred=prediction)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

loss_average = tf.keras.metrics.Mean()
accuracy = tf.keras.metrics.CategoricalAccuracy()
for epoch in range(epochs): 
    for image, label in dataset: 
        loss = apply_gradients(image, label)
        loss_average.update_state(loss)
        accuracy.update_state(label, model(image, training=False))
    print(f"epoch {epoch}: loss: {loss_average.result()} acc: {accuracy.result()}")
   
model.save('model.h5')
