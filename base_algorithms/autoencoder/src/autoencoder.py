##############
#  AE Class  #
##############

#%tensorflow_version 1.14

import csv
import pickle
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as py
import plotly.express as px
import random
import re
import sys
import tensorflow as tf
import time

#from google.colab.patches import cv2_imshow

from tensorflow.keras.layers import Input, Dense, Lambda, Flatten, Reshape, Conv2D, Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization, LeakyReLU, Activation, Dropout, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.datasets import mnist
from keras.layers import merge
from keras.utils import plot_model
from matplotlib import pyplot
from skimage.measure import compare_ssim as ssim
from tensorflow.keras import backend as K
from keras.objectives import binary_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from plotly import tools
from sklearn.manifold import TSNE


class ConvAutoencoder:

	def __init__(self, name):
		self.name = name

	def _build(self, height, width, depth, filters=(32, 64), latentDim=16):
		print("[INFO] Setting up the Autoencoder: Encoder + Decoder")
		# initialize the input shape to be "channels last" along with
		# the channels dimension itself
		# channels dimension itself
		inputShape = (height, width, depth)
		chanDim = -1

		# define the input to the encoder
		inputs = Input(shape=inputShape)
		x = inputs

		n_filters = len(filters)
		# loop over the number of filters
		for f in range(n_filters - 1):
			# apply a CONV => RELU => BN operation
			x = Conv2D(filters[f], (3, 3), strides=2, padding="same")(x)
			#x = MaxPooling2D((2, 2), padding='same')(x)
			x = LeakyReLU(alpha=0.2)(x)
			x = BatchNormalization(axis=chanDim)(x)
			x = Dropout(dropout_percent)(x)
		x = Conv2D(filters[n_filters - 1], (3, 3), strides=2, padding="same")(x)
		x = Activation("tanh")(x)
		x = BatchNormalization(axis=chanDim)(x)
		x = Dropout(dropout_percent)(x)

		# flatten the network and then construct our latent vector
		volumeSize = K.int_shape(x)
		x = Flatten()(x)
		latent = Dense(latentDim)(x)

		# build the encoder model
		encoder = Model(inputs, latent, name="encoder")

		print("[INFO] Encoder Summary")
		encoder.summary()

		# start building the decoder model which will accept the
		# output of the encoder as its inputs
		latentInputs = Input(shape=(latentDim,))
		x = Dense(np.prod(volumeSize[1:]))(latentInputs)
		x = Reshape((volumeSize[1], volumeSize[2], volumeSize[3]))(x)

		# loop over our number of filters again, but this time in
		# reverse order
		r_filters = filters[::-1]
		for f in range(n_filters - 1):
			# apply a CONV_TRANSPOSE => RELU => BN operation
			x = Conv2DTranspose(r_filters[f], (3, 3), strides=2, padding="same")(x)
			#x = UpSampling2D((2, 2))(x)
			x = LeakyReLU(alpha=0.2)(x)
			x = BatchNormalization(axis=chanDim)(x)
			x = Dropout(dropout_percent)(x)
		x = Conv2DTranspose(r_filters[n_filters - 1], (3, 3), strides=2, padding="same")(x)
		x = Activation("tanh")(x)
		x = BatchNormalization(axis=chanDim)(x)
		x = Dropout(dropout_percent)(x)


		# apply a single CONV_TRANSPOSE layer used to recover the
		# original depth of the image
		x = Conv2DTranspose(depth, (3, 3), padding="same")(x)
		outputs = Activation("sigmoid")(x)
		#outputs = Activation("tanh")(x)

		# build the decoder model
		decoder = Model(latentInputs, outputs, name="decoder")

		print("[INFO] Decoder Summary")
		decoder.summary()

		# our autoencoder is the encoder + decoder
		autoencoder = Model(inputs, decoder(encoder(inputs)),
			name="autoencoder")

		print("[INFO] Autoencoder Summary")
		autoencoder.summary()

		print("[INFO] Autoencoder configured!")

		# return a 3-tuple of the encoder, decoder, and autoencoder
		return (encoder, decoder, autoencoder)


	def gen_model(self, trainX, valX, testX, outlierX, train_step, val_step, test_step, outlier_step, image_height, image_width, image_depth, n_epochs=5, batch_size=32, init_LR=1e-3):

		# Get current epoch number
		path, dirs, files = next(os.walk(models_path))
		num_checkpoints = len(files)
		#n_epochs = (n_epochs - num_checkpoints)

		# construct our convolutional autoencoder
		print("[INFO] Building autoencoder...")
		#filters=[32, 64, 128, 256]
		filters = [32,64]
		latentDim=16
		#filters=[64, 128, 256, 512]
		#latentDim=32

		H = None

		reset_model = (num_checkpoints == 0)
		#reset_model = True

		if reset_model:
			(encoder, decoder, autoencoder) = self._build(image_height, image_width, image_depth, filters, latentDim)
			opt = Adam(lr=init_LR, decay=init_LR / n_epochs)
			#autoencoder.compile(loss="mae", optimizer=opt, metrics=['accuracy'])
			autoencoder.compile(loss=tf.keras.losses.MeanAbsoluteError(), optimizer=opt, metrics=['accuracy'])
			print("[INFO] Autoencoder compiled")

			# Install callbacks
			print("[INFO] Installing model checkpoint and TensorBoard callbacks")
			checkpointer = ModelCheckpoint(models_path + "ae_msc_coffee_epoch_{epoch:02d}-loss_{loss:.4f}-val_loss_{val_loss:.4f}.h5", monitor='loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
			tbCallBack = TensorBoard(log_dir=tensorboard_path, histogram_freq=0, write_graph=True, write_images=True, update_freq=600000)
			#earlystopper = EarlyStopping(monitor='val_loss', patience=15, verbose=1)
			callbacks_list = [checkpointer, tbCallBack] #, earlystopper]

			print("[INFO] Starting the Model training...")
			# train the convolutional autoencoder
			H = autoencoder.fit(x=trainX,
								steps_per_epoch=train_step,
								validation_data=valX,
								validation_steps=val_step,
								epochs=n_epochs,
								callbacks=callbacks_list)

			# Plot latrent spaces for training and testing datasets
			plot_latent_space(encoder, trainX, "Train", out_plot_file_latent_train)
			plot_latent_space(encoder, testX, "Test", out_plot_file_latent_test)
			plot_latent_space(encoder, valX, "Val", out_plot_file_latent_val)
			plot_latent_space(encoder, valX, "[Val + Outliers]", out_plot_file_latent_valoutlier, outlierX)

			# Printing current available metrics for the model
			print("[INFO] Model available metrics:")
			print(str(autoencoder.metrics_names))

		else:
			filepath = None
			str_num_checkpoints = str(num_checkpoints)
			if len(str_num_checkpoints) == 1:
				str_num_checkpoints = "0" + str_num_checkpoints
			ref_name = "ae_msc_coffee_epoch_" + str_num_checkpoints
			for filename in files:
				if ref_name in filename:
					filepath = models_path + filename
					print("[INFO] Loading model file: " + filepath)
					break
			if not filepath:
				raise(RuntimeError("No model file found"))
			autoencoder = load_model(filepath)
			#assert_allclose(self.model.predict(x_train), new_model.predict(x_train), 1e-5)

			# Install callbacks
			print("[INFO] Installing model checkpoint and TensorBoard callbacks")
			checkpointer = ModelCheckpoint(models_path + "ae_msc_coffee_epoch_" + str(num_checkpoints + 1) + "-loss_{loss:.4f}-val_loss_{val_loss:.4f}.h5", monitor='loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)
			tbCallBack = TensorBoard(log_dir=tensorboard_path, histogram_freq=0, write_graph=True, write_images=True, update_freq=600000)
			#earlystopper = EarlyStopping(monitor='val_loss', patience=15, verbose=1)
			callbacks_list = [checkpointer, tbCallBack] #, earlystopper]

			if (num_checkpoints <= n_epochs):
				print("[INFO] Starting the Model training...")
				H = autoencoder.fit(x=trainX,
									steps_per_epoch=train_step,
									validation_data=valX,
									validation_steps=val_step,
									epochs=(n_epochs - num_checkpoints),
									callbacks=callbacks_list)

		make_predictions(autoencoder,
						testX,
						trainX,
						outlierX,  
						recons_visual_file_test,
						recons_visual_file_train,
						recons_visual_file_outlier)

		plot_metrics(n_epochs, 
					out_plot_file_loss,
					out_plot_file_acc,
					H)
