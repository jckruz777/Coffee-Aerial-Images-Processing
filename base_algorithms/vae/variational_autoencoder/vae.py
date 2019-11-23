from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import matplotlib.pyplot as plt
import argparse
import cv2

from keras_preprocessing.image import ImageDataGenerator
from anomaly_detector import AnomalyDetector
from sklearn.metrics import roc_curve, auc
from vae_cnn_model import VAECNN
import random
import utils
import pandas as pd
import tensorflow as tf

import sys
sys.path.append('..')
import config
from imutils import paths

def create_csv_datasets(data_path, extension, train_csv_name, val_csv_name, test_csv_name):
    path_list = utils.get_subsections_paths(data_path, extension)
    list_train, list_val, list_test = utils.get_datapath_lists(path_list, 20, 3, 3)
    utils.create_csv(train_csv_name, list_train)
    utils.create_csv(val_csv_name, list_val)
    utils.create_csv(test_csv_name, list_test)
    totalTrain = len(list_train)
    totalVal = len(list_val)
    totalTest = len(list_test)
    return (totalTrain, totalVal, totalTest)

def main():
    parser = argparse.ArgumentParser()
    help_ = "Load h5 model trained weights"
    parser.add_argument("-w", "--weights", help=help_)
    help_ = "Use mse loss instead of binary cross entropy (default)"
    parser.add_argument("-m",
                        "--mse",
                        help=help_,action='store_true')
    help_ = "Epochs value (default = 50)"
    parser.add_argument("--epochs",
                        help=help_,
                        type=int,
                        default=50)
    help_ = "Batch value (default = 256)"
    parser.add_argument("--batch",
                        help=help_,
                        type=int,
                        default=32)
    help_ = "Use CNN VAE"
    parser.add_argument("--cnn",
                        help=help_,
                        action='store_true')
    help_ = "Test the breast cancer dataset"
    parser.add_argument("--test",
                        help=help_,
                        action='store_true')
    help_ = "Enable the plot feature"
    parser.add_argument("-p",
                        "--plot",
                        help=help_, action='store_true')
    help_ = "Predict image reconstruction"
    parser.add_argument("--predict",
                        help=help_,
                        default='')
    help_ = "Anomaly treshold value between 0 and 1 (Default is 0.45)"
    parser.add_argument("--anomaly-treshold",
                        help=help_,
                        type=float,
                        default=0.45)
    help_ = "Extension of the images to work with (Default is .JPG)"
    parser.add_argument("--ext",
                        default=".JPG",
                        help=help_)
    args = parser.parse_args()

    predict_img = str(args.predict)
    image_width = 100
    image_height = 100
    extension = str(args.ext)
    original_dim = image_width * image_height
    anomalyTreshold = float(args.anomaly_treshold)

    # .CSV file names
    train_csv_name = "./train_data.csv"
    val_csv_name = "./val_data.csv"
    test_csv_name = "./test_data.csv"

    # Get a list with the paths of all the images
    main_dataset_path = os.path.sep.join([config.NET_BASE, config.BASE_COFFEE_IMAGES]) + "/"
    print("Loading images from: " + main_dataset_path)
    #if not (os.path.exists(train_csv_name) and os.path.exists(val_csv_name) and os.path.exists(test_csv_name)):
    (totalTrain, totalVal, totalTest) = create_csv_datasets(main_dataset_path, extension, train_csv_name, val_csv_name, test_csv_name)

    # Load datasets from the created .CSV
    traindf = pd.read_csv(train_csv_name, dtype = str)
    valdf = pd.read_csv(val_csv_name, dtype = str)
    testdf = pd.read_csv(test_csv_name, dtype = str)

    # VAE model = encoder + decoder
    vae = None
    if args.cnn:
        inputShape = (image_width, image_height, 3)
        vae = VAECNN(input_shape=inputShape, latent_cont_dim=8, latent_disc_dim=3)
    else:
        inputShape = (image_width) * (image_height)
        vae = VAE(inputShape, args.batch, args.epochs, image_width, image_height)
        vae.build()

        # VAE loss = mse_loss or xent_loss + kl_loss
        reconstruction_loss = "mse" if args.mse else "binary_crossentropy"
        vae.setReconstructionError(reconstruction_loss)

        # Compile the model
        vae.compile()

    if args.weights:
        vae.loadWeights(args.weights)
    elif args.cnn:

        # train the autoencoder
        print("Loading the dataset...")

        # initialize the training data augmentation object
        trainAug = ImageDataGenerator(
        	rescale=1 / 255.0,
        	#rotation_range=20,
        	#zoom_range=0.05,
        	#width_shift_range=0.1,
        	#height_shift_range=0.1,
        	#shear_range=0.05,
        	#horizontal_flip=True,
        	#vertical_flip=True,
        	fill_mode="nearest")
            #preprocessing_function=imageResize)

        print("Training data augmentation object initialized")

        # initialize the validation (and testing) data augmentation object
        valAug = ImageDataGenerator(rescale=1 / 255.0)
                                    #preprocessing_function=imageResize)

        print("Validation and testing data augmentation object initialized")

        # initialize the training generator
        # genPath = config.TRAIN_COFFEE_PATH
        trainGen = trainAug.flow_from_dataframe(
                dataframe = traindf,
                directory = None,
                x_col = "image_path",
                y_col = None,
                class_mode="input",
                target_size=(image_width, image_height),
                batch_size=args.batch,
                color_mode="rgb",
                shuffle=True)

        print("Training generator initialized")

        # initialize the validation generator
        # genPath = config.VAL_COFFEE_PATH
        valGen = valAug.flow_from_dataframe(
                dataframe = valdf,
                directory = None,
                x_col = "image_path",
                y_col = None,
                class_mode="input",
                target_size=(image_width, image_height),
                batch_size=args.batch,
                color_mode="rgb",
                shuffle=True)

        print("Validation generator initialized")

        # initialize the testing generator
        # genPath = config.TEST_COFFEE_PATH
        testGen = valAug.flow_from_dataframe(
                dataframe = testdf,
        	directory = None,
                x_col = "image_path",
                y_col = None,
        	class_mode="input",
                target_size=(image_width, image_height),
        	batch_size=args.batch,
        	color_mode="rgb",
        	shuffle=True)

        print("Testing generator initialized")

        #x_train, x_val = utils.getData(nd_images=True)
        print("Dataset loaded")
        gpu_options = tf.compat.v1.GPUOptions(allow_growth = True)
        session = tf.compat.v1.InteractiveSession(config = tf.compat.v1.ConfigProto(gpu_options = gpu_options))
        print("Start training...")
        vae.fit(trainGen, valGen, num_epochs=args.epochs, batch_size=args.batch, totalTrain=totalTrain, totalVal=totalVal, totalTest=totalTest)

    if (args.plot):
        vae.plot()

    if predict_img != '':
        img = cv2.imread(predict_img)
        orig = img
        #img = utils.preprocess(img, image_width, image_height)
        #images = np.array([img])
        reconstruction_error, ssim, rec = vae.prediction(img)
        print("Anomaly treshold: " + str(anomalyTreshold))
        print("Reconstruction error: " + str(reconstruction_error))
        print("SSIM: " + str(ssim))

        detector = AnomalyDetector(anomaly_treshold = anomalyTreshold)
        detector.evaluate(reconstruction_error, ssim, orig, rec)

    if args.test:

        y_prob = []
        y_res = []
        normal_res = []
        anormal_res = []

        patients = os.listdir(os.path.sep.join([config.NET_BASE, config.ORIG_INPUT_CANCER_DATASET]))
        random.seed(3)
        random.shuffle(patients)

        for patient in patients:
            x_normal, x_anormal = utils.getValData(patient, image_width, image_height)

            y_prob += np.zeros(len(x_normal)).tolist() + np.ones(len(x_anormal)).tolist()
            for normal_img in x_normal:
                reconstruction_error, ssim, _ = vae.prediction(normal_img)
                normal_res.append(reconstruction_error)
                if reconstruction_error < (anomalyTreshold*100) and ssim > anomalyTreshold :
                    y_res.append(0)
                else:
                    y_res.append(1)

            for anormal_img in x_anormal:
                reconstruction_error, ssim,  _ = vae.prediction(anormal_img)
                anormal_res.append(reconstruction_error)
                if reconstruction_error < (anomalyTreshold*100) and ssim > anomalyTreshold :
                    y_res.append(0)
                else:
                    y_res.append(1)


        fpr, tpr, thresholds = roc_curve(y_prob, y_res)

        plt.plot(fpr, tpr, color='orange', label='ROC')
        plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend()
        plt.savefig('roc.png')
        plt.show()

        print("Average error for normal images: " + str(np.average(np.array(normal_res))))
        print("Average error for anomal images: " + str(np.average(np.array(anormal_res))))

        plt.scatter(range(len(normal_res)), normal_res)
        plt.scatter(range(len(anormal_res)), anormal_res)
        plt.title('Reconstruction test')
        plt.ylabel('Loss')
        plt.xlabel('Image')
        plt.legend(['Normal', 'Anomaly'], loc='upper right')
        plt.savefig('reconstruction_test.png')
        plt.show()

if __name__ == "__main__":
    main()
