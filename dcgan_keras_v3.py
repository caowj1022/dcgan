import numpy as np
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose
from keras.layers.core import Activation, Flatten
from keras.layers import Dense, Reshape
from keras.layers.normalization import BatchNormalization
from keras.datasets import mnist
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU
from keras.initializers import TruncatedNormal
from keras.initializers import RandomNormal

import matplotlib.pyplot as plt
from PIL import Image
import sys

#from keras.utils import plot_model

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

	def __init__(self, input_shape = (28, 28, 1), output_shape = (28, 28, 1), df_dim = 64, gf_dim = 64, z_dim = 100, batch_size = 128, epoch = 100):
		self.input_shape = input_shape
		self.output_shape = output_shape
		self.df_dim = df_dim
		self.gf_dim = gf_dim
		self.z_dim = z_dim
		self.batch_size = batch_size
		self.epoch = epoch

		self.build_model()

	def discriminator(self):

		discriminator = Sequential()

		discriminator.add(Conv2D(filters = self.df_dim, kernel_size = (5, 5), padding = 'same', input_shape = self.input_shape, kernel_initializer = TruncatedNormal(stddev = 0.02)))
		discriminator.add(LeakyReLU(alpha = 0.2))
#		discriminator.add(Activation('tanh'))
		discriminator.add(MaxPooling2D())

		discriminator.add(Conv2D(filters = self.df_dim * 2, kernel_size = (5, 5), kernel_initializer = TruncatedNormal(stddev = 0.02)))
#		discriminator.add(BatchNormalization())
		discriminator.add(LeakyReLU(alpha = 0.2))
#		discriminator.add(Activation('tanh'))
		discriminator.add(MaxPooling2D(pool_size = (2, 2)))

#		discriminator.add(Conv2D(filters = self.df_dim * 4, kernel_size = (5, 5), strides = (2, 2), padding = 'same'))
#		discriminator.add(BatchNormalization())
#		discriminator.add(LeakyReLU(alpha = 0.2))

		discriminator.add(Flatten())
		discriminator.add(Dense(units = 1024, kernel_initializer = RandomNormal(stddev = 0.02)))
#		discriminator.add(Activation('tanh'))
		discriminator.add(LeakyReLU(alpha = 0.2))
		discriminator.add(Dense(units = 1, kernel_initializer = RandomNormal(stddev = 0.02)))
		discriminator.add(Activation('sigmoid'))

		return discriminator

	def generator(self):
		h, w = self.output_shape[0], self.output_shape[1]
		h2, w2 = int(h / 2), int(w / 2)
		h4, w4 = int(h2 / 2), int(w2 / 2)
		h8, w8 = int(h4 / 2), int(w4 / 2)

		generator = Sequential()
		generator.add(Dense(input_dim = self.z_dim, units = 1024, kernel_initializer = RandomNormal(stddev = 0.02)))
		generator.add(Activation('tanh'))

		generator.add(Dense(self.gf_dim*2 * h4 * w4, kernel_initializer = RandomNormal(stddev = 0.02)))
		generator.add(BatchNormalization())
		generator.add(Activation('tanh'))

		generator.add(Reshape(target_shape = (h4, w4, self.gf_dim*2), input_shape = (self.gf_dim*2*h4*w4, )))
		generator.add(UpSampling2D(size = (2, 2)))
		generator.add(Conv2D(filters = self.gf_dim, kernel_size = (5, 5), padding = 'same', kernel_initializer = TruncatedNormal(stddev = 0.02)))
#		generator.add(Conv2DTranspose(filters = self.gf_dim, kernel_size = (5, 5), strides = (2, 2), padding = 'same'))
#		generator.add(BatchNormalization())
		generator.add(Activation('tanh'))

#		generator.add(Conv2DTranspose(filters = 1, kernel_size = (5, 5), strides = (2, 2), padding = 'same'))
		generator.add(UpSampling2D(size = (2, 2)))
		generator.add(Conv2D(filters = 1, kernel_size = (5, 5), padding = 'same', kernel_initializer = TruncatedNormal(stddev = 0.02)))
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
		(X_trn, Y_trn), (X_tst, Y_tst) = mnist.load_data()
		X_trn = (X_trn.astype(np.float32) - 127.5) / 127.5
		X_trn = X_trn[:, :, :, None]

		d_optim = Adam(lr = 0.0002, beta_1 = 0.5, epsilon = 1e-8)
		g_optim = Adam(lr = 0.0002, beta_1 = 0.5, epsilon = 1e-8)

		self.g.compile(loss = 'binary_crossentropy', optimizer = 'SGD')
		self.d_g.compile(loss = 'binary_crossentropy', optimizer = g_optim)
		self.d.trainable = True
		self.d.compile(loss = 'binary_crossentropy', optimizer = d_optim)


		for epoch in range(self.epoch):
			num_batch = int(X_trn.shape[0]/self.batch_size)
			for index in range(num_batch):
				z = np.random.uniform(-1, 1, size = (self.batch_size, 100))
				real_imag = X_trn[index*self.batch_size:(index+1)*self.batch_size]
				fake_imag = self.g.predict(z)
				if index % 50 == 0:
					image = combine_images(fake_imag)
					image = image * 127.5 + 127.5
					Image.fromarray(image.astype(np.uint8)).save(str(epoch)+"_"+str(index)+".png")

				X = np.concatenate((real_imag, fake_imag))
				Y = [1] * self.batch_size + [0] * self.batch_size
				d_loss = self.d.train_on_batch(X, Y)

				z = np.random.uniform(-1, 1, size = (self.batch_size, 100))
				self.d.trainable = False

				# optimize generator twice
				g_loss = self.d_g.train_on_batch(z, [1] * self.batch_size)
				g_loss = self.d_g.train_on_batch(z, [1] * self.batch_size)

				self.d.trainable = True
				print ("epoch [%d/%d] batch [%d/%d], d_loss: %.8f, g_loss: %.8f" % (epoch, self.epoch, index, num_batch, d_loss, g_loss))
				if index % 10 == 9:
					self.g.save_weights('generator', True)
					self.d.save_weights('discriminator', True)


def main():
	dcgan = DCGAN()
	dcgan.train()

if __name__ == "__main__":
	main()