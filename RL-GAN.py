#Check RL loss for a batch of batch_size



import tensorflow as tf
import os
import time

from matplotlib import pyplot as plt
from IPython import display
import numpy as np
from collections import deque



BUFFER_SIZE = 60000
BATCH_SIZE = 40
LAMBDA = 100
X=0

EPOCHS = 3

MEMORY = deque(maxlen=10000)


target_model=tf.keras.models.load_model('target_model')


(train_dataset, train_labels), (test_dataset, test_labels) = tf.keras.datasets.mnist.load_data()

train_dataset = train_dataset.reshape(train_dataset.shape[0], 28, 28, 1).astype('float32')
train_dataset = (train_dataset - 127.5) / 127.5
perm=np.random.permutation(train_dataset.shape[0])
train_dataset=train_dataset[perm]
train_labels=train_labels[perm]
train_labels = tf.keras.utils.to_categorical(train_labels, 10)
train_dataset=tf.data.Dataset.from_tensor_slices((train_dataset,train_labels))
train_dataset = train_dataset.batch(BATCH_SIZE)


test_dataset = test_dataset.reshape(test_dataset.shape[0], 28, 28, 1).astype('float32')
test_dataset = (test_dataset - 127.5) / 127.5
perm=np.random.permutation(test_dataset.shape[0])
test_dataset=test_dataset[perm]
test_labels=test_labels[perm]
test_labels = tf.keras.utils.to_categorical(test_labels, 10)
test_dataset=tf.data.Dataset.from_tensor_slices((test_dataset,test_labels))
# test_dataset = test_dataset.batch(BATCH_SIZE)

target_images=np.load("label.npy")
target_label = np.arange(10)
target_label=np.float32(target_label)

for i in range(2):
    target_images=np.vstack((target_images,target_images))
    target_label=np.hstack((target_label,target_label))

perm=np.random.permutation(target_images.shape[0])
target_images=target_images[perm]
target_label=target_label[perm]
target_label = tf.keras.utils.to_categorical(target_label, 10)
target_dataset=tf.data.Dataset.from_tensor_slices((target_images,target_label))
target_dataset = target_dataset.batch(BATCH_SIZE)


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



#********************************************************************************************


def Actor():
	#input = state (original image and target image)
	# 28*28*2 -> 28*28*1
	input_image = tf.keras.layers.Input(shape=[28,28,1])
	label_image = tf.keras.layers.Input(shape=[28,28,1])

	down_stack = [
		downsample(64, 4, apply_batchnorm=False), 
		downsample(128, 4), 
		downsample(256, 4,stride=1),
	]

	up_stack = [
		upsample(128, 4,stride=1),
		upsample(64, 4), 
	]

	initializer = tf.random_normal_initializer(0., 0.02)
	last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
        strides=2,
        padding='same',
        kernel_initializer=initializer,
        activation='tanh')
	
	x = tf.keras.layers.Concatenate()([input_image,label_image])
	for down in down_stack:
		x = down(x)
		
	for up in up_stack:
		x = up(x)
		
	x = last(x)
	
	return tf.keras.Model(inputs=[input_image,label_image], outputs=x)
	
	

actor = Actor()
tf.keras.utils.plot_model(actor, to_file='actor.png',show_shapes=True, dpi=64)


def Actor_loss(Q1):
    return tf.math.reduce_mean(-Q1)
    


actor_optimizer = tf.keras.optimizers.Adam(3e-4, beta_1=0.5)



#********************************************************************************************



#********************************************************************************************


def Critic():
  input1 = tf.keras.layers.Input(shape=[28,28,1])
  input2 = tf.keras.layers.Input(shape=[28,28,1])
  input3 = tf.keras.layers.Input(shape=[28,28,1])
  x = tf.keras.layers.Concatenate()([input1, input2, input3])

  #Q1
  q1 = tf.keras.layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same')(x)
  q1 = tf.keras.layers.ReLU()(q1)
  q1 = tf.keras.layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same')(q1)
  q1 = tf.keras.layers.ReLU()(q1)
  q1 = tf.keras.layers.Flatten()(q1)
  q1 = tf.keras.layers.Dense(512)(q1)
  q1 = tf.keras.layers.ReLU()(q1)
  q1 = tf.keras.layers.Dense(1)(q1)

  #q2
  q2 = tf.keras.layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same')(x)
  q2 = tf.keras.layers.ReLU()(q2)
  q2 = tf.keras.layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same')(q2)
  q2 = tf.keras.layers.ReLU()(q2)
  q2 = tf.keras.layers.Flatten()(q2)
  q2 = tf.keras.layers.Dense(512)(q2)
  q2 = tf.keras.layers.ReLU()(q2)
  q2 = tf.keras.layers.Dense(1)(q2)

  return tf.keras.Model(inputs=[input1, input2, input3], outputs=[q1, q2])
	
	



critic = Critic()
tf.keras.utils.plot_model(critic, to_file='critic.png',show_shapes=True, dpi=64)


loss_object_2 = tf.keras.losses.MeanSquaredError()

def Critic_loss(reward, current_Q1, current_Q2):
    return loss_object_2(current_Q1, reward) + loss_object_2(current_Q2, reward) 


critic_optimizer = tf.keras.optimizers.Adam(3e-4, beta_1=0.5)




#********************************************************************************************



#********************************************************************************************

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



def adv_loss(preds, labels, is_targeted):
    real = tf.math.reduce_sum(labels * preds, 1)
    other = tf.math.reduce_max((1 - labels) * preds - (labels * 10000), 1)
    if is_targeted:
        return tf.math.reduce_sum(tf.math.maximum(0.0, other - real)),tf.math.maximum(tf.zeros_like(other-real,dtype=tf.float32), other - real)
    return tf.math.reduce_sum(tf.maximum(0.0, real - other)),tf.math.maximum(tf.zeros_like(other-real,dtype=tf.float32), real-other)


def generator_loss(disc_generated_output, gen_output, target,pred,label):
  gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
  gan_loss_1 = loss_object_1(tf.ones_like(disc_generated_output), disc_generated_output)
  # mean absolute error
  l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
  l1_loss_1 = tf.keras.backend.mean(tf.abs(target - gen_output),axis=[1,2,3])
  l_adv, l_adv_1 = adv_loss(pred,label,True)
  total_gan_loss = l_adv + gan_loss + (LAMBDA * l1_loss)
  total_gan_loss_1 = l_adv_1 + gan_loss_1 + (LAMBDA * l1_loss_1)

  return total_gan_loss, total_gan_loss_1, gan_loss, l1_loss,l_adv

#********************************************************************************************


#********************************************************************************************

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



loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
loss_object_1 = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction = tf.keras.losses.Reduction.NONE)

def discriminator_loss(disc_real_output, disc_generated_output):
  real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

  generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

  total_disc_loss = real_loss + generated_loss

  return total_disc_loss


#********************************************************************************************

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)




checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 actor_optimizer=actor_optimizer,
                                 critic_optimizer=critic_optimizer,
                                 generator=generator,
                                 discriminator=discriminator,
                                 actor=actor,
                                 critic=critic)



def generate_images(model,test_input,t,a,b):

  A=np.random.randint(40)
  target = target_images[A]
  test_input=tf.reshape(test_input,[1,28,28,1])
  target=tf.reshape(target,[1,28,28,1])
  label_t = np.argmax(target_label[A])
  action = actor([test_input,target])

  prediction = model(action, training=True)
  pred_fake=np.argmax(target_model.predict(prediction))
  pred_orig=np.argmax(target_model.predict(test_input))
  t=np.argmax(t)
  # print(pred)
  # print(t)
  # pred=target_model.predict(prediction)
  # (loss, accuracy)=target_model.evaluate(prediction, t, batch_size = 128, verbose = 1)
  plt.figure(figsize=(15,15))
  display_list = [test_input, prediction, target]
  title = ['Input Image - ' + str(t)+' Predicted Original -'+str(pred_orig)  , 'Predicted Image - '+ str(pred_fake), ' Target Prediction - '+str(label_t)]
  # print("TEST Accuracy- ",accuracy)
  for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.title(title[i])
    # getting the pixel values between [0, 1] to plot it.
    plt.imshow(tf.reshape(display_list[i],[28,28]) * 0.5 + 0.5)
    plt.axis('off')
  if(a=='train'):
    plt.savefig('output_images/train_'+str(b)+'.png')
  elif(a=='test'):
    plt.savefig('output_images/test_'+str(b)+'.png')



import datetime
log_dir="logs/"

summary_writer = tf.summary.create_file_writer(
  log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

@tf.function
def train_step(dataset, target,epoch,target_model):
  for target_image , target_label in target.take(1):
    input_image=dataset[0]
    t=dataset[1]
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape, tf.GradientTape() as actor_tape, tf.GradientTape() as critic_tape:
        # find action using actor
        action = actor([input_image,target_image])
        action_modified = action + tf.random.normal([BATCH_SIZE,28,28,1],0,0.1)
        # input the action to generator
        gen_output = generator(action_modified, training=True)

        disc_real_output = discriminator(input_image, training=True)
        disc_generated_output = discriminator( gen_output, training=True)
        pred=gen_output
        for layer in target_model.layers:
            pred = layer(pred)
        
        
        correct_prediction = tf.math.equal(tf.math.argmax(pred, 1), tf.math.argmax(t, 1))
        accuracy = tf.math.reduce_mean(tf.cast(correct_prediction, "float"))
        tf.print("TRAIN Accuracy- ",accuracy)
        gen_total_loss, gen_total_loss_1, gen_gan_loss, gen_l1_loss,l_adv = generator_loss(disc_generated_output, gen_output, input_image,pred,target_label)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)
        Q1,Q2 = critic([input_image,target_image,action]) 
        # find the reward to update the RL agent.
        reward = -gen_total_loss_1
        # find actor loss
        actor_loss = Actor_loss(Q1)
        # find critic loss
        critic_loss = Critic_loss(reward, Q1, Q2)
        # update actor and critic both neeche
    generator_gradients = gen_tape.gradient(gen_total_loss,
                                            generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,
                                                discriminator.trainable_variables)
    actor_gradients = actor_tape.gradient(actor_loss,actor.trainable_variables)
    critic_gradients = critic_tape.gradient(critic_loss,critic.trainable_variables)
    generator_optimizer.apply_gradients(zip(generator_gradients,
                                            generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                discriminator.trainable_variables))
    actor_optimizer.apply_gradients(zip(actor_gradients,actor.trainable_variables))
    critic_optimizer.apply_gradients(zip(critic_gradients,critic.trainable_variables))


def fit(train_ds, target_ds, epochs, test_ds,target_model):
  for epoch in range(epochs):
    start = time.time()

    display.clear_output(wait=True)

    
    print("Epoch: ", epoch)
    # test_ds.shuffle(BUFFER_SIZE)
    for input_image,t in test_ds.take(1):
      generate_images(generator, input_image,t,'train',epoch)
    # Train
    n=0
    for  dataset in train_ds:
      # print(dataset)
      print('.', end='')
      if (n+1) % 100 == 0:
        print()
      train_step(dataset, target_ds, epoch,target_model)
      
      n+=1
    print()

    
    # saving (checkpoint) the model every 20 epochs
    if (epoch + 1) % 20 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                        time.time()-start))
  checkpoint.save(file_prefix = checkpoint_prefix)

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()

fit(train_dataset, target_dataset, EPOCHS, test_dataset,target_model)

# restoring the latest checkpoint in checkpoint_dir
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

"""## Generate using test dataset"""

# Run the trained model on a few examples from the test dataset
for input_image,t in test_dataset.take(100):
	generate_images(generator, input_image,t,'test',X)
	X+=1



# test_dataset = test_dataset.batch(10000)
# for dataset in test_dataset:
#   data=dataset[0]
#   t=dataset[1]
#   gen_output = generator(data, training=True)
#   pred=target_model.predict(gen_output)
#   correct_prediction = np.equal(np.argmax(pred, 1), np.argmax(t, 1))
#   accuracy = tf.math.reduce_mean(tf.cast(correct_prediction, "float"))
#   tf.print("acc:- ",accuracy)