import numpy as np
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose
from keras.layers.core import Activation, Flatten
from keras.layers import Dense, Reshape, Dropout
from keras.layers.normalization import BatchNormalization
from keras.datasets import mnist
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU
from keras.initializers import TruncatedNormal
from keras.initializers import RandomNormal

from keras import backend as k
import matplotlib.pyplot as plt
from PIL import Image
import sys

import struct

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

def combine_images_2(true_imag, fake_imag):
	assert len(true_imag) == len(fake_imag)
	num_images = true_imag.shape[0]
	manifold_h = int(np.floor(np.sqrt(num_images)))
	manifold_w = int(np.ceil(np.sqrt(num_images)))

	image = np.zeros((2* manifold_h * images.shape[1], 2* manifold_w * images.shape[2]), dtype = images.dtype)

	for index, img in enumerate(images):
		i = int(index/manifold_w)
		j = index % manifold_w
		image[i*images.shape[1]:(i+1)*images.shape[1], j*images.shape[2]:(j+1)*images.shape[2]] = img[:,:,0]

	return image	


class DCGAN(object):

	def __init__(self, input_shape = (28, 28, 1), output_shape = (28, 28, 1), df_dim = 64, gf_dim = 64, z_dim = 50, batch_size = 128, epoch = 100):
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

		# First layer
		# Input shape 28x28x1
		# Output shape 14x14x64
		discriminator.add(Conv2D(filters = self.df_dim, kernel_size = (5, 5), strides = (2, 2), padding = 'same', input_shape = self.input_shape, kernel_initializer = RandomNormal(stddev = 0.2)))
		discriminator.add(LeakyReLU(alpha = 0.2))
		discriminator.add(Dropout(0.3))

		# Second layer
		# Input shape 14x14x64
		# Output shape 7x7x128
		discriminator.add(Conv2D(filters = self.df_dim * 2, kernel_size = (5, 5), strides = (2, 2), padding = 'same'))
		discriminator.add(LeakyReLU(alpha = 0.2))
		discriminator.add(Dropout(0.3))
		
		# Third layer
		# Input Shape 7x7x128
		# Output Shape 1
		discriminator.add(Flatten())
		discriminator.add(Dense(units = 1))
		discriminator.add(Activation('sigmoid'))

		return discriminator

	def generator(self):
		h, w = self.output_shape[0], self.output_shape[1]
		h2, w2 = int(h / 2), int(w / 2)
		h4, w4 = int(h2 / 2), int(w2 / 2)

		generator = Sequential()

		# First layer
		# Input shape 100x1x1
		# Output shape 7x7x128
		generator.add(Dense(input_dim = self.z_dim, units = self.gf_dim*2 * h4 * w4, kernel_initializer = RandomNormal(stddev = 0.02)))
		generator.add(LeakyReLU(0.2))
		generator.add(Reshape(target_shape = (h4, w4, self.gf_dim*2), input_shape = (self.gf_dim*2*h4*w4, )))

		# Second layer
		# Input shape 7x7x128
		# Output shape 14x14x64
		generator.add(UpSampling2D(size = (2, 2)))
		generator.add(Conv2D(filters = self.gf_dim, kernel_size = (5, 5), padding = 'same'))
		generator.add((LeakyReLU(0.2)))

		# Third layer
		# Input shape 14x14x64
		# Output shape 28x28x1
		generator.add(UpSampling2D(size = (2, 2)))
		generator.add(Conv2D(filters = 1, kernel_size = (5, 5), padding = 'same'))
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

	def loss_function(self, Y_True, Y_Pred):
		return k.mean(k.square(Y_Pred - Y_True))

	def cross_correlation(self, Y_True, Y_Pred):
		return k.mean((Y_True - k.mean(Y_True))*(Y_Pred-k.mean(Y_Pred)))/(k.std(Y_True)*k.std(Y_Pred))

	def adam(self):
		pass

	def train(self):

		# Normalize Training Set
		(X_trn, Y_trn), (X_tst, Y_tst) = mnist.load_data()
		X_trn = (X_trn.astype(np.float32) - 127.5) / 127.5
		X_trn = X_trn[:, :, :, None]

		X_tst = (X_tst.astype(np.float32) - 127.5) / 127.5
		X_tst = X_tst[:, :, :, None]

		# Adam optimizer
		d_optim = Adam(lr = 0.0002, beta_1 = 0.5, epsilon = 1e-8)
		g_optim = Adam(lr = 0.0002, beta_1 = 0.5, epsilon = 1e-8)
		dg_optim = Adam(lr = 0.0002, beta_1 = 0.5, epsilon = 1e-8)


		# Binary crossentropy as loss function
		self.g.compile(loss = 'binary_crossentropy', optimizer = g_optim)
		self.d_g.compile(loss = 'binary_crossentropy', optimizer = dg_optim)
		self.d.trainable = True
		self.d.compile(loss = 'binary_crossentropy', optimizer = d_optim)

		self.g.load_weights('generator', True)
		self.d.load_weights('discriminator', True)

		input_tensor = self.g.layers[0].input
		output_tensor = self.g.layers[len(self.g.layers)-1].output
		y_true_tensor = k.placeholder(shape = (self.batch_size, 28, 28, 1), dtype = 'float32')
#		loss_tensor = self.cross_correlation(y_true_tensor, output_tensor)
		loss_tensor = self.loss_function(y_true_tensor, output_tensor)

		get_gradients = k.function([input_tensor, y_true_tensor], k.gradients(loss_tensor, input_tensor))
		get_loss = k.function([input_tensor, y_true_tensor], [loss_tensor])


		lr = 0.001
		beta1 = 0.9
		beta2 = 0.999
		epsilon = 1e-8
		iteration = 50000

		file = open('compressed.txt', 'wb')

		number = 32
		First_time = True
		for epoch in range(number):
			rd = np.random.random_integers(0, len(X_tst))
			tst_img = X_tst[rd:rd+1,:,:,:]
			z = np.random.uniform(-1, 1, size = (self.batch_size, self.z_dim))

			t = 0
			m_t = 0
			v_t = 0
			for i in range(iteration):
				gradients = get_gradients([z, tst_img])
				gradients = np.reshape(gradients, (self.batch_size, self.z_dim))
				t += 1
				lr_t = lr * np.sqrt((1-beta2**t)/(1-beta1**t))
				m_t = np.multiply(beta1,m_t) + np.multiply((1-beta1),gradients)
				v_t = np.multiply(beta2,v_t) + np.multiply((1-beta2), (np.multiply(gradients, gradients)))

				z = z - np.multiply(lr_t,m_t)/(np.sqrt(v_t)+epsilon)

#				gradients = np.reshape(gradients, (self.batch_size, self.z_dim))
#				change = learnging_rate*gradients
#				z = z - change

				# Stochastic clipping
				index = np.where(np.abs(z) > 1)
				for p in range(np.shape(index)[1]):
					m = index[0][p]
					n = index[1][p]
					z[m][n] = np.random.uniform(-1, 1)

				if i % 1000 == 0:
					loss = get_loss([z, tst_img])
					print ("epoch [%d], iteration [%d], loss [%.8f]" % (epoch, i, loss[0]))

			fake_imag = self.g.predict(z)

			for I in range(z.shape[0]):
				for J in range(z.shape[1]):
					file.write(struct.pack('f', z[I][J]))

			if First_time:
				First_time = False
				images = np.concatenate((tst_img, fake_imag), axis = 0)
			else:
				images = np.concatenate((images, tst_img, fake_imag), axis = 0)


		image = combine_images(np.array(images))
		image = image * 127.5 + 127.5
		Image.fromarray(image.astype(np.uint8)).save("test_adam/image1.png")



#############################
# Further Work:
# 1. find a better loss function, cross correlation is used now
# 2. Implement a better gradient decsent method, such as Adam
# 3. Try to solve local minimum or maximum problem



def main():

	dcgan = DCGAN(batch_size = 1)
	dcgan.train()

if __name__ == "__main__":
	main()