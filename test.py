import tensorflow as tf
import os
import time

from matplotlib import pyplot as plt
from IPython import display
import numpy as np




BUFFER_SIZE = 60000
BATCH_SIZE = 256
LAMBDA = 100
X=0

EPOCHS = 10



# target_model= tf.keras.Sequential([
#             tf.keras.layers.Conv2D(filters = 20,kernel_size = (5, 5),padding = "same",input_shape = (28, 28, 1),activation="relu"),
#             tf.keras.layers.MaxPool2D(pool_size = (2, 2),strides =  (2, 2)),
#             tf.keras.layers.Conv2D(filters = 50,kernel_size = (5, 5),padding = "same",activation="relu"),
#             tf.keras.layers.MaxPool2D(pool_size = (2, 2),strides =  (2, 2)),
#             tf.keras.layers.Flatten(),
#             tf.keras.layers.Dense(500,activation='relu', kernel_initializer="he_uniform"),
#             tf.keras.layers.Dense(10,activation='softmax', kernel_initializer="he_uniform")
#             ])
# target_model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),metrics = ["accuracy"])

target_model=tf.keras.models.load_model('target_model')


# IMG_WIDTH = 256
# IMG_HEIGHT = 256

# def load(image_file):
#   image = tf.io.read_file(image_file)
#   image = tf.image.decode_jpeg(image)

#   w = tf.shape(image)[1]

#   w = w // 2
#   real_image = image[:, :w, :]
#   input_image = image[:, w:, :]

#   input_image = tf.cast(input_image, tf.float32)
#   real_image = tf.cast(real_image, tf.float32)

#   return input_image, real_image

# inp, re = load(PATH+'train/100.jpg')
# # casting to int for matplotlib to show the image
# plt.figure()
# plt.imshow(inp/255.0)
# plt.figure()
# plt.imshow(re/255.0)

# def resize(input_image, real_image, height, width):
#   input_image = tf.image.resize(input_image, [height, width],
#                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
#   real_image = tf.image.resize(real_image, [height, width],
#                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

#   return input_image, real_image

# def random_crop(input_image, real_image):
#   stacked_image = tf.stack([input_image, real_image], axis=0)
#   cropped_image = tf.image.random_crop(
#       stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])

#   return cropped_image[0], cropped_image[1]


# def normalize(input_image, real_image):
#   input_image = (input_image / 127.5) - 1
#   real_image = (real_image / 127.5) - 1

#   return input_image, real_image

# @tf.function()
# def random_jitter(input_image, real_image):
#   # resizing to 286 x 286 x 3
#   input_image, real_image = resize(input_image, real_image, 286, 286)

#   # randomly cropping to 256 x 256 x 3
#   input_image, real_image = random_crop(input_image, real_image)

#   if tf.random.uniform(()) > 0.5:
#     # random mirroring
#     input_image = tf.image.flip_left_right(input_image)
#     real_image = tf.image.flip_left_right(real_image)

#   return input_image, real_image

# plt.figure(figsize=(6, 6))
# for i in range(4):
#   rj_inp, rj_re = random_jitter(inp, re)
#   plt.subplot(2, 2, i+1)
#   plt.imshow(rj_inp/255.0)
#   plt.axis('off')
# plt.show()

# def load_image_train(image_file):
#   input_image, real_image = load(image_file)
#   # input_image, real_image = random_jitter(input_image, real_image)
#   input_image, real_image = normalize(input_image, real_image)

#   return input_image, real_image

# def load_image_test(image_file):
#   input_image, real_image = load(image_file)
#   input_image, real_image = resize(input_image, real_image,
#                                    IMG_HEIGHT, IMG_WIDTH)
#   input_image, real_image = normalize(input_image, real_image)

#   return input_image, real_image

(train_dataset, train_labels), (test_dataset, test_labels) = tf.keras.datasets.mnist.load_data()

train_dataset = train_dataset.reshape(train_dataset.shape[0], 28, 28, 1).astype('float32')
train_dataset = (train_dataset - 127.5) / 127.5
perm=np.random.permutation(train_dataset.shape[0])
train_dataset=train_dataset[perm]
train_labels=train_labels[perm]
train_labels = tf.keras.utils.to_categorical(train_labels, 10)
train_dataset=tf.data.Dataset.from_tensor_slices((train_dataset,train_labels))
train_dataset = train_dataset.batch(BATCH_SIZE)

# test_dataset = tf.data.Dataset.list_files(PATH+'test/*.jpg')
# test_dataset = test_dataset.map(load_image_test)

test_dataset = test_dataset.reshape(test_dataset.shape[0], 28, 28, 1).astype('float32')
test_dataset = (test_dataset - 127.5) / 127.5
perm=np.random.permutation(test_dataset.shape[0])
test_dataset=test_dataset[perm]
test_labels=test_labels[perm]
test_labels = tf.keras.utils.to_categorical(test_labels, 10)
test_dataset=tf.data.Dataset.from_tensor_slices((test_dataset,test_labels))
# test_dataset = test_dataset.batch(BATCH_SIZE)


# print(train_dataset[1])

# print(test_dataset)
"""## Build the Generator
  * The architecture of generator is a modified U-Net.
  * Each block in the encoder is (Conv -> Batchnorm -> Leaky ReLU)
  * Each block in the decoder is (Transposed Conv -> Batchnorm -> Dropout(applied to the first 3 blocks) -> ReLU)
  * There are skip connections between the encoder and decoder (as in U-Net).
"""




OUTPUT_CHANNELS = 1

def downsample(filters, size, stride=2, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=stride, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result

# down_model = downsample(3, 4)
# down_result = down_model(tf.expand_dims(inp, 0))
# print (down_result.shape)

def upsample(filters, size, stride=2, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=stride,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

  result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.3))

  result.add(tf.keras.layers.ReLU())

  return result

# up_model = upsample(3, 4)
# up_result = up_model(down_result)
# print (up_result.shape)

def Generator():
  inputs = tf.keras.layers.Input(shape=[28,28,1])

  down_stack = [
    downsample(64, 4, apply_batchnorm=False), # (bs, 128, 128, 64)
    downsample(128, 4), # (bs, 64, 64, 128)
    downsample(256, 4,stride=1), # (bs, 32, 32, 256)
    # downsample(512, 4), # (bs, 16, 16, 512)
    # downsample(512, 4), # (bs, 8, 8, 512)
    # downsample(512, 4), # (bs, 4, 4, 512)
    # downsample(512, 4), # (bs, 2, 2, 512)
    # downsample(512, 4), # (bs, 1, 1, 512)
  ]

  up_stack = [
    # upsample(512, 4, apply_dropout=True), # (bs, 2, 2, 1024)
    # upsample(512, 4, apply_dropout=True), # (bs, 4, 4, 1024)
    # upsample(512, 4, apply_dropout=True), # (bs, 8, 8, 1024)
    # upsample(512, 4), # (bs, 16, 16, 1024)
    # upsample(256, 4), # (bs, 32, 32, 512)
    upsample(128, 4,stride=1), # (bs, 64, 64, 256)
    upsample(64, 4), # (bs, 128, 128, 128)
  ]

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh') # (bs, 256, 256, 3)

  x = inputs

  # Downsampling through the model
  skips = []
  for down in down_stack:
    x = down(x)
    skips.append(x)

  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    x = tf.keras.layers.Concatenate()([x, skip])

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)

generator = Generator()
tf.keras.utils.plot_model(generator, to_file='generator.png',show_shapes=True, dpi=64)

# gen_output = generator(inp[tf.newaxis,...], training=False)
# plt.imshow(gen_output[0,...])

"""* **Generator loss**
  * It is a sigmoid cross entropy loss of the generated images and an **array of ones**.
  * The [paper](https://arxiv.org/abs/1611.07004) also includes L1 loss which is MAE (mean absolute error) between the generated image and the target image.
  * This allows the generated image to become structurally similar to the target image.
  * The formula to calculate the total generator loss = gan_loss + LAMBDA * l1_loss, where LAMBDA = 100. This value was decided by the authors of the [paper](https://arxiv.org/abs/1611.07004).

The training procedure for the generator is shown below:
"""

def adv_loss(preds, labels, is_targeted):
	real = tf.math.reduce_sum(labels * preds, 1)
	other = tf.math.reduce_max((1 - labels) * preds - (labels * 10000), 1)
	if is_targeted:
		return tf.math.reduce_sum(tf.math.maximum(0.0, other - real))
	return tf.math.reduce_sum(tf.maximum(0.0, real - other))


def generator_loss(disc_generated_output, gen_output, target,pred,label):
  gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

  # mean absolute error
  l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
  l_adv = adv_loss(pred,label,False)
  total_gan_loss = l_adv + gan_loss + (LAMBDA * l1_loss)
  

  return total_gan_loss, gan_loss, l1_loss,l_adv

"""![Generator Update Image](https://github.com/tensorflow/docs/blob/master/site/en/tutorials/generative/images/gen.png?raw=1)

## Build the Discriminator
  * The Discriminator is a PatchGAN.
  * Each block in the discriminator is (Conv -> BatchNorm -> Leaky ReLU)
  * The shape of the output after the last layer is (batch_size, 30, 30, 1)
  * Each 30x30 patch of the output classifies a 70x70 portion of the input image (such an architecture is called a PatchGAN).
  * Discriminator receives 2 inputs.
    * Input image and the target image, which it should classify as real.
    * Input image and the generated image (output of generator), which it should classify as fake.
    * We concatenate these 2 inputs together in the code (`tf.concat([inp, tar], axis=-1)`)
"""

def Discriminator():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[28, 28, 1]))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1))

    return model

discriminator = Discriminator()
tf.keras.utils.plot_model(discriminator, to_file='disriminator.png',show_shapes=True, dpi=64)

# disc_out = discriminator([inp[tf.newaxis,...], gen_output], training=False)
# plt.imshow(disc_out[0,...,-1], vmin=-20, vmax=20, cmap='RdBu_r')
# plt.colorbar()

"""**Discriminator loss**
  * The discriminator loss function takes 2 inputs; **real images, generated images**
  * real_loss is a sigmoid cross entropy loss of the **real images** and an **array of ones(since these are the real images)**
  * generated_loss is a sigmoid cross entropy loss of the **generated images** and an **array of zeros(since these are the fake images)**
  * Then the total_loss is the sum of real_loss and the generated_loss
"""

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(disc_real_output, disc_generated_output):
  real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

  generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

  total_disc_loss = real_loss + generated_loss

  return total_disc_loss





"""The training procedure for the discriminator is shown below.

To learn more about the architecture and the hyperparameters you can refer the [paper](https://arxiv.org/abs/1611.07004).

![Discriminator Update Image](https://github.com/tensorflow/docs/blob/master/site/en/tutorials/generative/images/dis.png?raw=1)

## Define the Optimizers and Checkpoint-saver
"""
generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

"""## Generate Images

Write a function to plot some images during training.

* We pass images from the test dataset to the generator.
* The generator will then translate the input image into the output.
* Last step is to plot the predictions and **voila!**

Note: The `training=True` is intentional here since
we want the batch statistics while running the model
on the test dataset. If we use training=False, we will get
the accumulated statistics learned from the training dataset
(which we don't want)
"""

def generate_images(model,test_input,t,a,b):
  test_input=tf.reshape(test_input,[1,28,28,1])
  prediction = model(test_input, training=True)
  pred_fake=np.argmax(target_model.predict(prediction))
  pred_orig=np.argmax(target_model.predict(test_input))
  t=np.argmax(t)
  # print(pred)
  # print(t)
  # pred=target_model.predict(prediction)
  # (loss, accuracy)=target_model.evaluate(prediction, t, batch_size = 128, verbose = 1)
  plt.figure(figsize=(15,15))
  display_list = [test_input, prediction]
  title = ['Input Image - ' + str(t)+' Predicted Original -'+str(pred_orig) , 'Predicted Image - '+ str(pred_fake)]
  # print("TEST Accuracy- ",accuracy)
  for i in range(2):
    plt.subplot(1, 2, i+1)
    plt.title(title[i])
    # getting the pixel values between [0, 1] to plot it.
    plt.imshow(tf.reshape(display_list[i],[28,28]) * 0.5 + 0.5)
    plt.axis('off')
  if(a=='train'):
    plt.savefig('output_images/train_'+str(b)+'.png')
  elif(a=='test'):
    plt.savefig('output_images/test_'+str(b)+'.png')

# for example_input in test_dataset.take(1):
# 	generate_images(generator, example_input)

"""## Training

* For each example input generate an output.
* The discriminator receives the input_image and the generated image as the first input. The second input is the input_image and the target_image.
* Next, we calculate the generator and the discriminator loss.
* Then, we calculate the gradients of loss with respect to both the generator and the discriminator variables(inputs) and apply those to the optimizer.
* Then log the losses to TensorBoard.
"""


import datetime
log_dir="logs/"

summary_writer = tf.summary.create_file_writer(
  log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

@tf.function
def train_step(dataset, epoch,target_model):
  input_image=dataset[0]
  t=dataset[1]
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    gen_output = generator(input_image, training=True)

    disc_real_output = discriminator(input_image, training=True)
    disc_generated_output = discriminator( gen_output, training=True)
    pred=gen_output
    for layer in target_model.layers:
    	pred = layer(pred)
    # pred=target_model.predict(gen_output,steps=1)
    # (loss, accuracy)=target_model.evaluate(gen_output, t, batch_size = 128, verbose = 1,steps=1)
    correct_prediction = tf.math.equal(tf.math.argmax(pred, 1), tf.math.argmax(t, 1))
    accuracy = tf.math.reduce_mean(tf.cast(correct_prediction, "float"))

    gen_total_loss, gen_gan_loss, gen_l1_loss,l_adv = generator_loss(disc_generated_output, gen_output, input_image,pred,t)
    disc_loss = discriminator_loss(disc_real_output, disc_generated_output)
  print("TRAIN Accuracy- ",accuracy)
  generator_gradients = gen_tape.gradient(gen_total_loss,
                                          generator.trainable_variables)
  discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)

  generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
  discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))


  # with summary_writer.as_default():
  #   tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
  #   tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
  #   tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)
  #   tf.summary.scalar('disc_loss', disc_loss, step=epoch)

"""The actual training loop:

* Iterates over the number of epochs.
* On each epoch it clears the display, and runs `generate_images` to show it's progress.
* On each epoch it iterates over the training dataset, printing a '.' for each example.
* It saves a checkpoint every 20 epochs.
"""

def fit(train_ds, epochs, test_ds,target_model):
  for epoch in range(epochs):
    start = time.time()

    display.clear_output(wait=True)

    
    print("Epoch: ", epoch)
    for input_image,t in test_ds.take(1):
      generate_images(generator, input_image,t,'train',epoch)
    # Train
    n=0
    for  dataset in train_ds:
      # print(dataset)
      print('.', end='')
      if (n+1) % 100 == 0:
        print()
      train_step(dataset, epoch,target_model)
      
      n+=1
    print()

    
    # saving (checkpoint) the model every 20 epochs
    if (epoch + 1) % 20 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                        time.time()-start))
  checkpoint.save(file_prefix = checkpoint_prefix)



"""This training loop saves logs you can easily view in TensorBoard to monitor the training progress. Working locally you would launch a separate tensorboard process. In a notebook, if you want to monitor with TensorBoard it's easiest to launch the viewer before starting the training.

To launch the viewer paste the following into a code-cell:
"""

# Commented out IPython magic to ensure Python compatibility.
#docs_infra: no_execute
# %load_ext tensorboard
# %tensorboard --logdir {log_dir}

"""Now run the training loop:"""

# fit(train_dataset, EPOCHS, test_dataset,target_model)

# restoring the latest checkpoint in checkpoint_dir
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

"""## Generate using test dataset"""

# Run the trained model on a few examples from the test dataset
for input_image,t in test_dataset.take(30):
	generate_images(generator, input_image,t,'test',X)
	X+=1