import numpy as np
import tensorflow as tf
import os
import time
from tensorflow.keras.applications.resnet import *
from tensorflow.python.keras import backend
from tensorflow.python.keras import layers
from tensorflow.python.keras.engine import training
# model = tf.keras.applications.ResNet50V2(include_top=False, input_shape=(32,32,3), pooling='avg')

# (train_dataset, train_labels), (test_dataset, test_labels) = tf.keras.datasets.cifar10.load_data()

# print(model.summary())

def BasicBlock(x, filters, kernel_size=3, stride=1, conv_shortcut=False, name=None):
  """A residual block.
  Arguments:
      x: input tensor.
      filters: integer, filters of the bottleneck layer.
      kernel_size: default 3, kernel size of the bottleneck layer.
      stride: default 1, stride of the first layer.
      conv_shortcut: default False, use convolution shortcut if True,
        otherwise identity shortcut.
      name: string, block label.
  Returns:
    Output tensor for the residual block.
  """
  bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

  if conv_shortcut:
    shortcut = layers.Conv2D(
        filters, 1, strides=stride, name=name + '_0_conv')(x)
  else:
    shortcut = layers.MaxPooling2D(1, strides=stride)(x) if stride > 1 else x

  x = layers.Conv2D(
      filters, kernel_size=3, strides=stride, use_bias=False, padding='same', name=name + '_1_conv')(x)
  x = layers.BatchNormalization(
      axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x)
  x = layers.Activation('relu', name=name + '_1_relu')(x)

  # x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name=name + '_2_pad')(x)
  x = layers.Conv2D(
      filters,
      kernel_size=3,
      strides=1,
      use_bias=False,
      padding='same',
      name=name + '_2_conv')(x)
  x = layers.BatchNormalization(
      axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn')(x)
  x = layers.Add(name=name + '_out')([shortcut, x])
  x = layers.Activation('relu', name=name + '_2_relu')(x)

  return x


def stack(x, filters, num_blocks, stride, name=None):
  strides = [1]*(num_blocks-1)
  x = BasicBlock(x, filters, stride=stride, conv_shortcut=True, name=name+'_block1')
  for i in range(len(strides)):
    x = BasicBlock(x, filters, stride=strides[i], name=name+'_block'+str(i+2))
  return x

def ResNet56V2_stack_fn(x):
  bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
  x = layers.Conv2D(
      16, kernel_size=3, strides=1, use_bias=False, padding='same', name='conv1')(x)
  x = layers.BatchNormalization(
      axis=bn_axis, epsilon=1.001e-5, name='bn1')(x)
  x = layers.Activation('relu', name='relu1')(x)
  x = stack(x, 16, 9, stride=1, name='conv2')
  x = stack(x, 32, 9, stride=2, name='conv3')
  x = stack(x, 64, 9, stride=2, name='conv4')
  x = tf.keras.layers.AveragePooling2D(8, name='pool1')(x)
  # x = tf.keras.backend.flatten(x)
  x = tf.keras.layers.Dense(10, activation='softmax' , name='predictions')(x)
  return x

def ResNet56V2(input_tensor=None):
  input_shape = (32, 32, 3)
  if input_tensor is None:
    img_input = layers.Input(shape=input_shape)
  else:
    if not backend.is_keras_tensor(input_tensor):
      img_input = layers.Input(tensor=input_tensor, shape=input_shape)
      
    else:
      img_input = input_tensor

  x = ResNet56V2_stack_fn(img_input)
  if input_tensor is not None:
    inputs = layer_utils.get_source_inputs(input_tensor)
  else:
    inputs = img_input
  model = training.Model(inputs, x, name='ResNet56V2')
  return model
