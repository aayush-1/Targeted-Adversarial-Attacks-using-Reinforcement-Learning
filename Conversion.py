# Load the CIFAR10 data.
(x, y_train), (x_t, y_test) = cifar10.load_data()

rgb_weights = [0.2989, 0.5870, 0.1140]
x_train = []
for i in range(50000):
  grayscale_image = np.dot(x[i][...,:3], rgb_weights)
  #np.reshape(grayscale_image,(32,32,1))
  x_train.append(grayscale_image)

x_test = []
for i in range(10000):
  grayscale_image = np.dot(x_t[i][...,:3], rgb_weights)
  #np.reshape(grayscale_image,(32,32,1))
  x_test.append(grayscale_image)

x_train = np.array(x_train)
x_test = np.array(x_test)

x_train = np.reshape(x_train,(50000,32,32,1))
x_test = np.reshape(x_test,(10000,32,32,1))
