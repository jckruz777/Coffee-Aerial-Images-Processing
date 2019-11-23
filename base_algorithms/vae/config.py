# import the necessary packages
import os

# network base path
NET_BASE = os.path.sep.join(["..", "..", "..", "..", ".."])

# initialize the path to the *original* input directory of images
ORIG_INPUT_COFFEE_DATASET = "coffee_dataset"

# base path from the images will listed
BASE_COFFEE_IMAGES = os.path.sep.join([ORIG_INPUT_COFFEE_DATASET, "ICAFÉ_proyecto"])

# initialize the base path to the *new* directory that will contain
# our images after computing the training and testing split
BASE_COFFEE_PATH = os.path.sep.join([ORIG_INPUT_COFFEE_DATASET, "datasets/coffee"])

# derive the training, validation, and testing directories
TRAIN_COFFEE_PATH = os.path.sep.join([BASE_COFFEE_PATH, "training"])
VAL_COFFEE_PATH = os.path.sep.join([BASE_COFFEE_PATH, "validation"])
TEST_COFFEE_PATH = os.path.sep.join([BASE_COFFEE_PATH, "testing"])

#ELLIPS_NORMAL = os.path.sep.join([NET_BASE, ORIG_INPUT_ELLIPS_DATASET, "no_anomalies"])
#ELLIPS_ANORMAL = os.path.sep.join([NET_BASE, ORIG_INPUT_ELLIPS_DATASET, "anomalies"])

# define the amount of data that will be used training
TRAIN_SPLIT = 0.8

# the amount of validation data will be a percentage of the *training* data
VAL_SPLIT = 0.1
