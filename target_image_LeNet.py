
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# Download the MNIST dataset
(x_train, y_train), (x_test, y_test)= tf.keras.datasets.mnist.load_data()
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

target_model=tf.keras.models.load_model('target_model')

# Evaluate the model
# (loss, accuracy) = model.evaluate(x_test, y_test, batch_size = 128, verbose = 1)
# model.save("target_model")
# print(accuracy)

list_prob=[0,0,0,0,0,0,0,0,0,0]

list_images=[]
for i in range(10):
    list_images.append(np.zeros_like(x_train[0]))
C=np.ones((10,28,28,1))
print(x_train.shape)

for i in range(400):
    A=target_model.predict(x_train[i].reshape(1,28,28,1))
    A=A[0]
    print(x_train.shape)
    # print(i)
    # print(A)
    t=np.argmax(A)
    if(A[t]>list_prob[t]):
        C[t,:,:,:]=np.array(x_train[i]).reshape(28,28,1)
        list_prob[t]=A[t]
        list_images[t]=x_train[i]
        print("yess")



for i in range(10):
    C[i]=list_images[i].reshape(28,28,1)
    plt.imshow(list_images[i].reshape(28,28))
    plt.savefig("labels/"+str(i)+".png")


np.save("label",C)


S=np.load("label.npy")

print(S.shape)