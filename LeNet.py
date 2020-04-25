
from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy as np
import tensorflow as tf

# Download the MNIST dataset
(x_train, y_train), (x_test, y_test)= tf.keras.datasets.mnist.load_data(path='mnist.npz')
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
print(np.max(x_train))


x_train=x_train/255
x_test=x_test/255

# # Reshape the data to a (70000, 28, 28, 1) tensord
x_train = x_train[:, :, :, np.newaxis]
x_test = x_test[:, :, :, np.newaxis]



# Tranform training labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, 10)

# Tranform test labels to one-hot encoding
y_test = tf.keras.utils.to_categorical(y_test, 10)



model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters = 20,kernel_size = (5, 5),padding = "same",input_shape = (28, 28, 1),activation="relu"),
            tf.keras.layers.MaxPool2D(pool_size = (2, 2),strides =  (2, 2)),
            tf.keras.layers.Conv2D(filters = 50,kernel_size = (5, 5),padding = "same",activation="relu"),
            tf.keras.layers.MaxPool2D(pool_size = (2, 2),strides =  (2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(500,activation='relu', kernel_initializer="he_uniform"),
            tf.keras.layers.Dense(10,activation='softmax', kernel_initializer="he_uniform")
            ])
model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),metrics = ["accuracy"])






# Train the model 
model.fit(x_train,y_train, batch_size = 128, epochs = 20, verbose = 1)

# Evaluate the model
(loss, accuracy) = model.evaluate(x_test, y_test, batch_size = 128, verbose = 1)

# print(accuracy)