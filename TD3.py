import numpy as np
import tensorflow as tf

"""
Trick One: Clipped Double-Q Learning. TD3 learns two Q-functions instead of one (hence “twin”), and uses the smaller of the two Q-values to form the targets in the Bellman error loss functions.

Trick Two: “Delayed” Policy Updates. TD3 updates the policy (and target networks) less frequently than the Q-function. The paper recommends one policy update for every two Q-function updates.

Trick Three: Target Policy Smoothing. TD3 adds noise to the target action, to make it harder for the policy to exploit Q-function errors by smoothing out Q along changes in action.

abe critic start kar
usme mostly output size 1 hai
discriminator jesa bnana hai
vha se copy past krke start ho jaa


"""

OUTPUT_CHANNELS = 1

DISCOUNT=0.99,
TAU=0.005,
POLICY_NOISE=0.2,
NOISE_CLIP=0.5,
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
  
  x = tf.keras.layers.Concatenate(input_image,label_image)
  for down in down_stack:
    x = down(x)
    
  for up in up_stack:
    x = up(x)
    
  x = last(x)
  
  return tf.keras.Model(inputs=[input_image,label_image], outputs=x)
  
  

actor = Actor()
tf.keras.utils.plot_model(actor, to_file='actor.png',show_shapes=True, dpi=64)

actor_target = Actor()
actor_target.set_weights(actor.get_weights())


actor_optimizer = tf.keras.optimizers.Adam(3e-4, beta_1=0.5)


def Actor_loss():
  Q1,_ = critic()



def Critic():
  #input = state and action both
  #input_size  =  28*28*3
  #output_size = bhai tu sequential mein 3 input ko concatanate 
  
  
  input1 = tf.keras.layers.Input(shape=[28,28,2])
	input2 = tf.keras.layers.Input(shape=[28,28,1])
  
  downstack = [
    downsample(64, 4, st)
  ]
  model.add(tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[28, 28, ]))
  model.add(tf.keras.layers.LeakyReLU())
  model.add(tf.keras.layers.Dropout(0.3))

  model.add(tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
  model.add(tf.keras.layers.LeakyReLU())
  model.add(tf.keras.layers.Dropout(0.3))

  model.add(tf.keras.layers.Flatten())
  model.add(tf.keras.layers.Dense(1))



critic_optimizer = tf.keras.optimizers.Adam(3e-4, beta_1=0.5)


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)

		# Q2 architecture
		self.l4 = nn.Linear(state_dim + action_dim, 256)
		self.l5 = nn.Linear(256, 256)
		self.l6 = nn.Linear(256, 1)


	def forward(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2
    

  
  
  
  
  
  
  
  
  
  
  
  
  