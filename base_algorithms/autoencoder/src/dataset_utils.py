# COFFEE

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

def get_subsections_paths(root_path, extension): # root_path = ICAFÉ_proyecto/
    """
    Returns a list with all the images paths sorted randomly
    """
    subsections_list = []
    subsections_outliers_list = []
    for root, dirs, files in os.walk(root_path):
        for name in files:
            file_abs_path = os.path.join(root, name)
            if name.endswith((extension)):
                if "Outliers" in file_abs_path: 
                    subsections_outliers_list.append([file_abs_path])
                else:
                    subsections_list.append([file_abs_path])
    random.shuffle(subsections_list)
    random.shuffle(subsections_list)
    random.shuffle(subsections_outliers_list)
    random.shuffle(subsections_outliers_list)
    return (subsections_list, subsections_outliers_list)


def get_datapath_lists(subsections_list, outliers_list, train_percent, val_percent, test_percent, batch_size):
    """
    Returns three lists with the train, validation and testing image paths
    """
    random.shuffle(subsections_list)
    random.shuffle(outliers_list)
    list_size = len(subsections_list)
    list_outliers_size = len(outliers_list)
    n_train = round(train_percent * (list_size / 100.0))
    n_val = round(val_percent / 2.0 * (list_size / 100.0))
    n_test = round(test_percent * (list_size / 100.0))
    n_train = get_divisible(n_train, batch_size)
    n_val = get_divisible(n_val, batch_size)
    n_test = get_divisible(n_test, batch_size)
    print("\n================================================================================")
    print("Total Images: " + str(list_size))
    print("Total Used Images: " + str(n_train + n_test + n_val))
    print("Training Images: " + str(n_train))
    print("Validate Images: " + str(n_val))
    print("Testing Images: " + str(n_test))
    print("Outlier Images: " + str(n_val))
    print("================================================================================\n")
    list_train = subsections_list[0 : n_train]
    list_val = subsections_list[n_train : (n_train + n_val)]
    list_test = subsections_list[(n_train + n_val) : (n_train + n_val + n_test)]
    list_outlier = outliers_list[0 : n_val]
    return (list_train, list_val, list_test, list_outlier)


def create_csv(csv_path, column_list):
    with open(csv_path, 'w', newline='') as csv_file:
        wr = csv.writer(csv_file, quoting = csv.QUOTE_ALL)
        wr.writerow(["image_path"])
        for path in column_list:
            wr.writerow(path)
    csv_file.close()


def create_csv_datasets(data_path, extension, train_csv_name, val_csv_name, test_csv_name, outlier_csv_name, batch_size, gen_files=True):
    (path_list, outliers_list) = get_subsections_paths(data_path, extension)
    list_train, list_val, list_test, list_outliers = get_datapath_lists(
        path_list, 
        outliers_list, 
        train_percent, 
        val_percent, 
        test_percent, 
        batch_size)
    if gen_files:
        create_csv(train_csv_name, list_train)
        create_csv(val_csv_name, list_val)
        create_csv(outlier_csv_name, list_outliers)
        create_csv(test_csv_name, list_test)
    totalTrain = len(list_train)
    totalVal = len(list_val)
    totalTest = len(list_test)
    totalOutliers = len(list_outliers)
    return (totalTrain, totalVal, totalTest, totalOutliers)


def load_coffee_dataset(batch_size):
    # Get a list with the paths of all the images
    print("[INFO] Loading images from: " + coffee_dataset_path)
    if not (os.path.exists(train_csv_name) and os.path.exists(val_csv_name) and os.path.exists(test_csv_name)):
        (totalTrain, totalVal, totalTest, totalOutliers) = create_csv_datasets(
            coffee_dataset_path, 
            extension, 
            train_csv_name, 
            val_csv_name, 
            test_csv_name,
            outlier_csv_name,
            batch_size, 
            True)
    else:
        (totalTrain, totalVal, totalTest, totalOutliers) = create_csv_datasets(
            coffee_dataset_path, 
            extension, 
            train_csv_name, 
            val_csv_name, 
            test_csv_name,
            outlier_csv_name,
            batch_size, 
            False)
    # Load datasets from the created .CSV
    traindf = pd.read_csv(train_csv_name, dtype = str)
    valdf = pd.read_csv(val_csv_name, dtype = str)
    testdf = pd.read_csv(test_csv_name, dtype = str)
    outlierdf = pd.read_csv(outlier_csv_name, dtype = str)
    return traindf, valdf, testdf, outlierdf, totalTrain, totalVal, totalTest, totalOutliers


def get_divisible(n_data, batch_size):
    """
    Produces an integer from the division of n_data by batch_size
    """
    if n_data > batch_size:
        mod_result = (n_data % batch_size)
        #print("Module: " + str(n_data) + " % " + str(batch_size) + " = " + str(n_data - mod_result))
        return (n_data - mod_result)

def augment_dataset(rescale, rotation_range, zoom_range, width_shift_range, height_shift_range, 
                    shear_range, horizontal_flip, vertical_flip, fill_mode):
    """
    Image Data Generator for augmentation
    """
    data_aug = ImageDataGenerator(
            featurewise_center=True,
            samplewise_center = True,
            rescale=rescale,
            #featurewise_std_normalization=True,
            rotation_range=rotation_range,
        	zoom_range=zoom_range,
        	width_shift_range=width_shift_range,
        	height_shift_range=height_shift_range,
        	shear_range=shear_range,
        	horizontal_flip=horizontal_flip,
        	vertical_flip=vertical_flip,
        	fill_mode=fill_mode)
    return data_aug

def load_dataframe(dataframe, augmented_props, directory, x_col, y_col, class_mode, \
                   target_height, target_width, batch_size, color_mode, shuffle):
    """
    Load images from a dataframe
    """
    loaded_df = augmented_props.flow_from_dataframe(
                dataframe = dataframe,
                directory = directory,
                x_col = x_col,
                y_col = y_col,
                class_mode=class_mode,
                target_size=(target_width, target_height),
                batch_size=batch_size,
                color_mode=color_mode,
                shuffle=shuffle)
    step_size = loaded_df.n//loaded_df.batch_size
    return step_size, loaded_df

def read_single_image(img_path, height, width):
        with open(img_path, 'rb') as f:
            img = Image.open(f)
            conv_img = img.convert('RGB').resize((width, height))
            np_array = (np.array(conv_img) * (1.0 / 255.0)).astype('float32')
            img.close()
            return np_array

def load_all_images(path_list, height, width):
    N_paths = len(path_list)
    if N_paths > 5500:
        N_paths = 5500
    result_array = []
    img_path = ""
    for path_n in range(N_paths):
        img_path = path_list[path_n][0]
        img_bytes = read_single_image(img_path, height, width)
        result_array.append(img_bytes)
    return result_array
