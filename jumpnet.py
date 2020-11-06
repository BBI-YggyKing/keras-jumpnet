# AICamp Deep Learning for Computer Vision (Cohort #5) - Capstone Assignment
# Yossarian King / October 2020

# Jumpnet implementation, based on https://github.com/GoogleCloudPlatform/keras-idiomatic-programmer/tree/master/zoo/jumpnet
#
# The implementation is in the idiomatic style - stem > learner > classifier.
# The learner consists of groups, each composed of blocks.
#
# For clarity of implementation, rather than parameterize all the options and feed them down
# the call stack, most parameters are hardwired into the stages of the implementation.
# This is less flexible, but is readable and explicit about the layers being created.
#
# Chosen settings precisely mimic the model built by jumpnet_c.py (see "Example of JumpNet for CIFAR-10")
#
# Model is trained on CIFAR-10 dataset, from https://www.cs.toronto.edu/~kriz/cifar.html
#
#
# Development:
# python 3.7 with tensorflow packages
# CUDA 10.1
# cuDNN 7.6.5 for CUDA 10.1

from tensorflow.keras import Input, Model
import tensorflow.keras.layers as layers

class JumpNet():

	def __init__(self, shape):
		self.inputs = Input(shape)
		self.layers = None
		self.model = None

	def stem(self, filters1=16, filters2=32, stride1=1, stride2=1):
		self.layers = layers.Conv2D(filters1, (3, 3), strides=stride1, padding='same', use_bias=False)(self.inputs)
		self.layers = layers.BatchNormalization()(self.layers)
		self.layers = layers.ReLU()(self.layers)

		self.layers = layers.Conv2D(filters2, (3, 3), strides=stride2, padding='same', use_bias=False)(self.layers)
		self.layers = layers.BatchNormalization()(self.layers)
		self.layers = layers.ReLU()(self.layers)
		return self

	def group(self, filters, blocks, blockfilters=None):
		shortcut = layers.BatchNormalization()(self.layers)
		shortcut = layers.Conv2D(filters, (1,1), strides=(2,2), use_bias=False)(shortcut)

		for _ in range(blocks):
			self.block(filters, blockfilters)

		self.layers = layers.BatchNormalization()(self.layers)
		self.layers = layers.ReLU()(self.layers)
		self.layers = layers.Conv2D(filters, (1,1), strides=(2,2), use_bias=False)(self.layers)

		self.layers = layers.Concatenate()([shortcut, self.layers])
		return self

	def block(self, filters, blockfilters=None):
		shortcut = self.layers

		blockfilters = blockfilters or filters
		self.layers = layers.BatchNormalization()(self.layers)
		self.layers = layers.ReLU()(self.layers)
		self.layers = layers.Conv2D(blockfilters, (1,1), strides=(1,1), use_bias=False)(self.layers)

		self.layers = layers.BatchNormalization()(self.layers)
		self.layers = layers.ReLU()(self.layers)
		self.layers = layers.Conv2D(blockfilters, (3,3), strides=(1,1), padding='same', use_bias=False)(self.layers)

		self.layers = layers.BatchNormalization()(self.layers)
		self.layers = layers.ReLU()(self.layers)
		self.layers = layers.Conv2D(filters, (1,1), strides=(1,1), use_bias=False)(self.layers)

		self.layers = layers.Add()([shortcut, self.layers])
		return self
	
	def classifier(self, classes):
		self.layers = layers.GlobalAveragePooling2D()(self.layers)
		self.layers = layers.Dense(classes)(self.layers)
		self.layers = layers.Activation('softmax')(self.layers)
		self.outputs = self.layers
		self.model = Model(self.inputs, self.outputs)
		return self
