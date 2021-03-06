import numpy as np

from keras.layers import Input, Dense, Lambda, Flatten, Reshape, merge
from keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.layers.core import Dropout
from keras.optimizers import RMSprop
from keras import backend as K
from keras.utils import plot_model
from keras.objectives import binary_crossentropy, mse
from utils import (kl_normal, kl_discrete, sampling_normal,
                  sampling_concrete, EPSILON)

import os
import cv2
import sys
sys.path.append('..')
import config
from imutils import paths
from skimage.measure import compare_ssim as ssim
import matplotlib.pyplot as plt


class VAECNN():
    """
    Class to handle building and training VAE models.
    """
    def __init__(self, input_shape=(48, 48, 3), latent_cont_dim=2,
                 latent_disc_dim=0, hidden_dim=128, filters=(64, 64, 64)):
        """
        Setting up everything.
        Parameters
        ----------
        input_shape : Array-like, shape (num_rows, num_cols, num_channels)
            Shape of image.
        latent_cont_dim : int
            Dimension of latent distribution.
        latent_disc_dim : int
            Dimension of discrete latent distribution.
        hidden_dim : int
            Dimension of hidden layer.
        filters : Array-like, shape (num_filters, num_filters, num_filters)
            Number of filters for each convolution in increasing order of
            depth.
        """
        self.opt = None
        self.model = None
        self.input_shape = input_shape
        self.latent_cont_dim = latent_cont_dim
        self.latent_disc_dim = latent_disc_dim
        self.latent_dim = self.latent_cont_dim + self.latent_disc_dim
        self.hidden_dim = hidden_dim
        self.filters = filters

    def fit(self, trainGen, valGen, num_epochs=1, batch_size=100, val_split=.1,
            learning_rate=1e-3, reset_model=True, totalTrain=0, totalVal=0, totalTest=0):
        """
        Training model
        """
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        if reset_model:
            self._set_model()

        if totalTrain % batch_size != 0:
            raise(RuntimeError("Training data shape {} is not divisible by batch size {}".format(totalTrain, self.batch_size)))

        # Update parameters
        #K.set_value(self.opt.lr, learning_rate)
        self.model.compile(optimizer=self.opt, loss=self._vae_loss)

        self._history = self.model.fit_generator(
	        trainGen,
        	steps_per_epoch=totalTrain // self.batch_size,
        	validation_data=valGen,
        	validation_steps=totalVal // self.batch_size,
        	epochs=self.num_epochs)

        self.model.save_weights('vae_cnn.h5')

    def loadWeights(self, weights, batch_size=1):
        self.batch_size = batch_size
        self._set_model()
        self.model.load_weights(weights)

    def _set_model(self):
        """
        Setup model (method should only be called in self.fit())
        """
        print("Setting up model...")
        chanDim = -1
        # Encoder
        inputs = Input(batch_shape=(self.batch_size,) + self.input_shape)

        # Instantiate encoder layers
        Q_0 = Conv2D(self.input_shape[2], (2, 2), padding='same',
                     activation='relu')
        Qn_0 = BatchNormalization(axis=chanDim)

        Q_1 = Conv2D(self.filters[0], (2, 2), padding='same', strides=(2, 2),
                     activation='relu')
        Qn_1 = BatchNormalization(axis=chanDim)

        Q_2 = Conv2D(self.filters[1], (3, 3), padding='same', strides=(1, 1),
                     activation='relu')
        Qn_2 = BatchNormalization(axis=chanDim)

        Q_3 = Conv2D(self.filters[2], (3, 3), padding='same', strides=(1, 1),
                     activation='relu')
        Qn_3 = BatchNormalization(axis=chanDim)

        Q_4 = Flatten()
        Q_5 = Dense(self.hidden_dim, activation='relu')
        Q_z_mean = Dense(self.latent_cont_dim)
        Q_z_log_var = Dense(self.latent_cont_dim)

        # Set up encoder
        x = Q_0(inputs)
        x = Qn_0(x)
        x = Q_1(x)
        x = Qn_1(x)
        x = Q_2(x)
        x = Qn_2(x)
        x = Q_3(x)
        x = Qn_3(x)
        flat = Q_4(x)
        hidden = Q_5(flat)
        hidden = BatchNormalization(axis=chanDim)(hidden)
        hidden = Dropout(0.5)(hidden)

        # Parameters for continous latent distribution
        z_mean = Q_z_mean(hidden)
        z_log_var = Q_z_log_var(hidden)
        # Parameters for concrete latent distribution
        if self.latent_disc_dim:
            Q_c = Dense(self.latent_disc_dim, activation='softmax')
            alpha = Q_c(hidden)

        # Sample from latent distributions
        if self.latent_disc_dim:
            print("Concrete distribution")
            z = Lambda(self._sampling_normal)([z_mean, z_log_var])
            c = Lambda(self._sampling_concrete)(alpha)
            encoding = Concatenate()([z, c])
        else:
            print("Normal distribution")
            encoding = Lambda(self._sampling_normal)([z_mean, z_log_var])

        # Generator
        # Instantiate generator layers to be able to sample from latent
        # distribution later
        out_shape = (int(self.input_shape[0] / 2), int(self.input_shape[1] / 2), self.filters[2])
        G_0 = Dense(self.hidden_dim, activation='relu')
        G_1 = Dense(np.prod(out_shape), activation='relu')
        G_2 = Reshape(out_shape)

        G_3 = Conv2DTranspose(self.filters[2], (3, 3), padding='same',
                              strides=(1, 1), activation='relu')
        Gn_3 = BatchNormalization(axis=chanDim)

        G_4 = Conv2DTranspose(self.filters[1], (3, 3), padding='same',
                              strides=(1, 1), activation='relu')
        Gn_4 = BatchNormalization(axis=chanDim)

        G_5 = Conv2DTranspose(self.filters[0], (2, 2), padding='valid',
                              strides=(2, 2), activation='relu')
        Gn_5 = BatchNormalization(axis=chanDim)

        G_6 = Conv2D(self.input_shape[2], (2, 2), padding='same',
                     strides=(1, 1), activation='sigmoid', name='generated')
        Gn_6 = BatchNormalization(axis=chanDim)

        # Apply generator layers
        x = G_0(encoding)
        x = BatchNormalization(axis=chanDim)(x)
        x = Dropout(0.5)(x)
        x = G_1(x)
        x = G_2(x)
        x = G_3(x)
        x = Gn_3(x)
        x = G_4(x)
        x = Gn_4(x)
        x = G_5(x)
        x = Gn_5(x)
        x = G_6(x)
        generated = Gn_6(x)

        self.model = Model(inputs, generated)

        # Set up generator
        inputs_G = Input(batch_shape=(self.batch_size, self.latent_dim))
        x = G_0(inputs_G)
        x = G_1(x)
        x = G_2(x)
        x = G_3(x)
        x = G_4(x)
        x = G_5(x)
        generated_G = G_6(x)
        self.generator = Model(inputs_G, generated_G)

        # Store latent distribution parameters
        self.z_mean = z_mean
        self.z_log_var = z_log_var
        if self.latent_disc_dim:
            self.alpha = alpha

        # Compile models
        self.opt = 'adam'
        self.model.compile(optimizer=self.opt, loss=self._vae_loss)
        # Loss and optimizer do not matter here as we do not train these models
        self.generator.compile(optimizer=self.opt, loss='mse')

        #self._plot(inputs, generated, self.model)

        print("Completed model setup.")

    def generate(self, latent_sample):
        """
        Generating examples from samples from the latent distribution.
        """
        # Model requires batch_size batches, so tile if this is not the case
        if latent_sample.shape[0] != self.batch_size:
            latent_sample = np.tile(latent_sample, self.batch_size).reshape(
                              (self.batch_size, self.latent_dim))
        return self.generator.predict(latent_sample, batch_size=self.batch_size)

    def _vae_loss(self, x, x_generated):
        """
        Variational Auto Encoder loss.
        """
        x = K.flatten(x)
        x_generated = K.flatten(x_generated)
        reconstruction_loss = self.input_shape[0] * self.input_shape[1] * \
                                  mse(x, x_generated)
        kl_normal_loss = kl_normal(self.z_mean, self.z_log_var)
        if self.latent_disc_dim:
            kl_disc_loss = kl_discrete(self.alpha)
        else:
            kl_disc_loss = 0
        return reconstruction_loss + kl_normal_loss + kl_disc_loss

    def _sampling_normal(self, args):
        """
        Sampling from a normal distribution.
        """
        z_mean, z_log_var = args
        return sampling_normal(z_mean, z_log_var, (self.batch_size, self.latent_cont_dim))

    def _sampling_concrete(self, args):
        """
        Sampling from a concrete distribution
        """
        return sampling_concrete(args, (self.batch_size, self.latent_disc_dim))

    def prediction(self, orig):
        img = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self._input_shape[0], self._input_shape[1]))
        orig = img
        img = img.astype('float32') / 255

        images = np.array([img])

        rec = self._vae.predict(images)
        rec = rec * 255
        rec = rec.astype('int32')

        loss = self._vae.evaluate(images, verbose=0)
        ssimg = ssim(orig, rec[0], multichannel=True)
        rec_img = rec[0]
        return (loss, ssimg, rec_img)

    def _plot(self, model):
        plot_model(model, to_file='vae_cnn.png', show_shapes=True)

    def plot(self):
        plt.plot(self._history.history['loss'])
        plt.plot(self._history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig('vae_cnn_error.png')
