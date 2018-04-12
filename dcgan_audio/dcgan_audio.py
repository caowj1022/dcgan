from scipy.io import wavfile
import os
import numpy as np
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, Conv1D, UpSampling1D
from keras.layers.core import Activation, Flatten
from keras.layers import Dense, Reshape, Dropout, ZeroPadding1D, Cropping1D
from keras.layers.normalization import BatchNormalization
from keras.datasets import mnist, cifar10
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU
from keras.initializers import TruncatedNormal
from keras.initializers import RandomNormal

from keras.utils import plot_model

import matplotlib.pyplot as plt
from PIL import Image
import sys

#np.set_printoptions(threshold=np.nan)

def combine_images(images):
	num_images = images.shape[0]
	manifold_h = int(np.floor(np.sqrt(num_images)))
	manifold_w = int(np.ceil(np.sqrt(num_images)))

	image = np.zeros((manifold_h * images.shape[1], manifold_w * images.shape[2]), dtype = images.dtype)

	for index, img in enumerate(images):
		i = int(index/manifold_w)
		j = index % manifold_w
		image[i*images.shape[1]:(i+1)*images.shape[1], j*images.shape[2]:(j+1)*images.shape[2]] = img[:,:,0]

	return image

class DCGAN(object):

	def __init__(self, trn_X, df_dim = 64, gf_dim = 64, z_dim = 100, batch_size = 64, epoch = 50):
		
		self.trn_X = trn_X
		self.input_shape = self.trn_X.shape[1:3]
		self.output_shape = self.input_shape
		self.df_dim = df_dim
		self.gf_dim = gf_dim
		self.z_dim = z_dim
		self.batch_size = batch_size
		self.epoch = epoch

		self.build_model()

	def discriminator(self):

		discriminator = Sequential()

		# 132299 + 1434 + 1435 = 135168
		# Input: 132299 * 2
		# Output: 135168 * 2
		discriminator.add(ZeroPadding1D((1434, 1435), input_shape = self.input_shape))

		# Input: 135168 * 2
		# Output: 33792 * d
		discriminator.add(Conv1D(filters = self.df_dim, kernel_size = 25, strides = 4, padding = 'same', kernel_initializer = RandomNormal(stddev = 0.2)))
		discriminator.add(LeakyReLU(alpha = 0.2))
		discriminator.add(Dropout(0.3))

		# Input: 33792 * d
		# Output: 8448 * 2d
		discriminator.add(Conv1D(filters = self.df_dim * 2, kernel_size = 25, strides = 4, padding = 'same'))
		discriminator.add(LeakyReLU(alpha = 0.2))
		discriminator.add(Dropout(0.3))

		# Input: 8448 * 2d
		# Output: 2112 * 4d
		discriminator.add(Conv1D(filters = self.df_dim * 4, kernel_size = 25, strides = 4, padding = 'same'))
		discriminator.add(LeakyReLU(alpha = 0.2))
		discriminator.add(Dropout(0.3))

		# Input: 2112 * 4d
		# Output: 528 * 8d
		discriminator.add(Conv1D(filters = self.df_dim * 8, kernel_size = 25, strides = 4, padding = 'same'))
		discriminator.add(LeakyReLU(alpha = 0.2))
		discriminator.add(Dropout(0.3))
		
		# Input: 528 * 8d
		# Output: 132 * 16d
		discriminator.add(Conv1D(filters = self.df_dim * 16, kernel_size = 25, strides = 4, padding = 'same'))
		discriminator.add(LeakyReLU(alpha = 0.2))
		discriminator.add(Dropout(0.3))

		# Input: 132 * 16d
		# Output: 33 * 32d
		discriminator.add(Conv1D(filters = self.df_dim * 32, kernel_size = 25, strides = 4, padding = 'same'))
		discriminator.add(LeakyReLU(alpha = 0.2))
		discriminator.add(Dropout(0.3))
		
		discriminator.add(Flatten())
		discriminator.add(Dense(units = 1))
		discriminator.add(Activation('sigmoid'))

		return discriminator

	def generator(self):

		generator = Sequential()

		# Input: 100 * 1
		# Output: 33 * 32d
		generator.add(Dense(input_dim = self.z_dim, units = self.gf_dim*32 * 33, kernel_initializer = RandomNormal(stddev = 0.02)))
		generator.add(LeakyReLU(0.2))
		generator.add(Reshape(target_shape = (33, self.gf_dim*32), input_shape = (self.gf_dim*32*33, )))

		# Input: 33 * 32d
		# Output: 132 * 16d
		generator.add(UpSampling1D(size = 4))
		generator.add(Conv1D(filters = self.gf_dim*16, kernel_size = 25, padding = 'same'))
		generator.add((LeakyReLU(0.2)))

		# Input: 132 * 16d
		# Output: 528 * 8d
		generator.add(UpSampling1D(size = 4))
		generator.add(Conv1D(filters = self.gf_dim*8, kernel_size = 25, padding = 'same'))
		generator.add((LeakyReLU(0.2)))

		# Input: 528 * 8d
		# Output: 2112 * 4d
		generator.add(UpSampling1D(size = 4))
		generator.add(Conv1D(filters = self.gf_dim*4, kernel_size = 25, padding = 'same'))
		generator.add((LeakyReLU(0.2)))

		# Input: 2112 * 4d
		# Output: 8448 * 2d
		generator.add(UpSampling1D(size = 4))
		generator.add(Conv1D(filters = self.gf_dim*2, kernel_size = 25, padding = 'same'))
		generator.add((LeakyReLU(0.2)))

		# Input: 8448 * 2d
		# Output: 33792 * d
		generator.add(UpSampling1D(size = 4))
		generator.add(Conv1D(filters = self.gf_dim, kernel_size = 25, padding = 'same'))
		generator.add((LeakyReLU(0.2)))

		# Input: 33792 * d
		# Output: 135168 * 2
		generator.add(UpSampling1D(size = 4))
		generator.add(Conv1D(filters = self.output_shape[1], kernel_size = 25, padding = 'same'))
		generator.add((LeakyReLU(0.2)))

		# Input: 135168 * 2
		# Output: 132299 * 2
		generator.add(Cropping1D((1434, 1435)))

		generator.add(Activation('tanh'))

		return generator

	def discriminate_generator(self, g, d):
		d_g = Sequential()
		d_g.add(g)
		d.trainable = False
		d_g.add(d)

		return d_g

	def build_model(self):

		self.d = self.discriminator()
		self.g = self.generator()
		self.d_g = self.discriminate_generator(self.g, self.d)

#		plot_model(self.d, to_file = 'discriminator.png', show_shapes = True)
#		plot_model(self.g, to_file = 'generator.png', show_shapes = True)
#		plot_model(self.d_g, to_file = 'discriminator(gen).png', show_shapes = True)

	def train(self):

		# Normalize Training Set
#		(X_trn, Y_trn), (X_tst, Y_tst) = mnist.load_data()
#		X_trn = (X_trn.astype(np.float32) - 127.5) / 127.5
#		X_trn = X_trn[:, :, :, None]

		X_trn = self.trn_X

		# Adam optimizer
		d_optim = Adam(lr = 0.0002, beta_1 = 0.5, epsilon = 1e-8)
		g_optim = Adam(lr = 0.0002, beta_1 = 0.5, epsilon = 1e-8)
		dg_optim = Adam(lr = 0.0002, beta_1 = 0.5, epsilon = 1e-8)

		# Binary crossentropy as loss function
		self.g.compile(loss = 'binary_crossentropy', optimizer = g_optim)
		self.d_g.compile(loss = 'binary_crossentropy', optimizer = dg_optim)
		self.d.trainable = True
		self.d.compile(loss = 'binary_crossentropy', optimizer = d_optim)

		for epoch in range(self.epoch):
			num_batch = int(X_trn.shape[0]/self.batch_size)
			bits_per_sample = 16
			maximum = 1<<(bits_per_sample-1)

			for index in range(num_batch):
				# Noise input
				z = np.random.uniform(-1, 1, size = (self.batch_size, 100))
				real_audio = X_trn[index*self.batch_size:(index+1)*self.batch_size]
				fake_audio = self.g.predict(z)
				if index % 50 == 0:
					fake_audio = fake_audio * maximum
					for i in range(self.batch_size):
						wavfile.write(str(epoch)+"_"+str(index)+"_"+str(i)+".wav", 44100, fake_audio[i,:,:].astype(np.int16))
				X = np.concatenate((real_audio, fake_audio))
				Y = [1] * self.batch_size + [0] * self.batch_size
				d_loss = self.d.train_on_batch(X, Y)

				z = np.random.uniform(-1, 1, size = (self.batch_size, 100))
				self.d.trainable = False

				# optimize generator twice
				g_loss = self.d_g.train_on_batch(z, [1] * self.batch_size)
				g_loss = self.d_g.train_on_batch(z, [1] * self.batch_size)

				self.d.trainable = True
				print ("epoch [%d/%d] batch [%d/%d], d_loss: %.8f, g_loss: %.8f" % (epoch, self.epoch, index, num_batch, d_loss, g_loss))
				if index % 10 == 0:
					self.g.save_weights('generator', True)
					self.d.save_weights('discriminator', True)

def main():

	trn_X = []
	bits_per_sample = 16

	path = os.getcwd()
	path = os.path.join(path, 'vio')

	for filename in os.listdir(path):
		file_path = os.path.join('vio', filename)

		fs, data = wavfile.read(file_path)
		trn_X.append(data)

	trn_X = np.array(trn_X)
	maximum = 1<<(bits_per_sample-1)
	trn_X = trn_X.astype(np.float32)/maximum

	dcgan = DCGAN(trn_X)
	dcgan.train()

if __name__ == "__main__":
	main()