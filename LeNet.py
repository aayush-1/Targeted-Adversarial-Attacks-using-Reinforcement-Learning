
from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy as np
import tensorflow as tf

# Download the MNIST dataset
dataset = datasets.fetch_mldata("MNIST Original")

# Reshape the data to a (70000, 28, 28) tensor
data = dataset.data.reshape((dataset.data.shape[0], 28, 28))

# Reshape the data to a (70000, 28, 28, 1) tensord
data = data[:, :, :, np.newaxis]

# Scale values from range of [0-255] to [0-1]
scaled_data = data / 255.0

# Split the dataset into training and test sets
(train_data, test_data, train_labels, test_labels) = train_test_split(
    scaled_data,
    dataset.target.astype("int"), 
    test_size = 0.33)

# Tranform training labels to one-hot encoding
train_labels = np_utils.to_categorical(train_labels, 10)

# Tranform test labels to one-hot encoding
test_labels = np_utils.to_categorical(test_labels, 10)



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
model.fit(train_data,train_labels, batch_size = 128, epochs = 20, verbose = 1)

# Evaluate the model
(loss, accuracy) = model.evaluate(test_data, test_labels, batch_size = 128, verbose = 1)

print(accuracy)