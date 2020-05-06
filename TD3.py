import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

"""
Trick One: Clipped Double-Q Learning. TD3 learns two Q-functions instead of one (hence “twin”), and uses the smaller of the two Q-values to form the targets in the Bellman error loss functions.

Trick Two: “Delayed” Policy Updates. TD3 updates the policy (and target networks) less frequently than the Q-function. The paper recommends one policy update for every two Q-function updates.

Trick Three: Target Policy Smoothing. TD3 adds noise to the target action, to make it harder for the policy to exploit Q-function errors by smoothing out Q along changes in action

"""

OUTPUT_CHANNELS = 1

POLICY_FREQ=2


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
	
	x = tf.keras.layers.Concatenate([input_image,label_image])
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


def Critic():
  input1 = tf.keras.layers.Input(shape=[28,28,1])
  input2 = tf.keras.layers.Input(shape=[28,28,1])
  input3 = tf.keras.layers.Input(shape=[28,28,1])
  x = tf.keras.layers.Concatenate([input1, input2, input3])

  #Q1
  q1 = keras.layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same')(x)
  q1 = keras.layers.ReLU()(q1)
  q1 = keras.layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same')(q1)
  q1 = keras.layers.ReLU()(q1)#
  q1 = keras.layers.Flatten()(q1)
  q1 = keras.layers.Dense(512)(q1)
  q1 = keras.layers.ReLU()(q1)
  q1 = keras.layers.Dense(1)(q1)

  #q2
  q2 = keras.layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same')(x)
  q2 = keras.layers.ReLU()(q2)
  q2 = keras.layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same')(q2)
  q2 = keras.layers.ReLU()(q2)
  q2 = keras.layers.Flatten()(q2)
  q2 = keras.layers.Dense(512)(q2)
  q1 = keras.layers.ReLU()(q1)
  q2 = keras.layers.Dense(1)(q2)

  return tf.keras.Model(inputs=[input1, input2, input3], outputs=[q1, q2])
	
	



critic = Critic()
tf.keras.utils.plot_model(critic, to_file='critic.png',show_shapes=True, dpi=64)

loss_object = tf.keras.losses.MeanSquaredError()


def Critic_loss(reward, current_Q1, current_Q2):
    return loss_object(current_Q1, reward) + loss_object(current_Q2, reward) 


critic_optimizer = tf.keras.optimizers.Adam(3e-4, beta_1=0.5)



	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
