#!/usr/bin/env python

import argparse

exec(open("constants.py").read())
exec(open("plot_utils.py").read())
exec(open("dataset_utils.py").read())
exec(open("autoencoder.py").read())

N_EPOCHS = 50
BATCH_SIZE = 64
INIT_LEARNING_RATE = 1e-3

def check_dirs():
	print("\n[INFO] Checking elemental paths...")
	elemental_paths = [models_path, tensorboard_path, csv_path, res_path]
	for elem_path in elemental_paths:
		if not os.path.exists(elem_path):
			os.makedirs(elem_path)

def parse_cli():
	print("[INFO] Parsing CLI arguments...")
	parser = argparse.ArgumentParser()
	help_ = "Epochs value (default = 50)"
	parser.add_argument("--epochs", help=help_, type=int, default=50)
	help_ = "Batch value (default = 32)"
	parser.add_argument("--batch", help=help_, type=int, default=32)
	help_ = "Initial Learning Rate value (default = 1e-3)"
	parser.add_argument("--init_lr", help=help_, type=float, default=1e-3)
	args = parser.parse_args()
	N_EPOCHS = int(args.epochs)
	BATCH_SIZE = int(args.batch)
	INIT_LEARNING_RATE = float(args.init_lr)

def build_dataset():

	# Loading the Coffee Dataset
	(traindf, valdf, testdf, outlierdf, totalTrain, totalVal, totalTest, totalOutliers) = load_coffee_dataset(BATCH_SIZE)

	# Data Augmentation
	augmented_dataset_train = augment_dataset(
	    rescale= 2.0 / 255.0, rotation_range = 10, zoom_range = 0.02, width_shift_range = 0.1,
	    height_shift_range = 0.1, shear_range = 0.01, horizontal_flip = True, vertical_flip = True,
	    fill_mode = "nearest")

	augmented_dataset_validate = augment_dataset(
	    rescale= 2.0 / 255.0, rotation_range = 10, zoom_range = 0.02, width_shift_range = 0.1,
	    height_shift_range = 0.1, shear_range = 0.01, horizontal_flip = True, vertical_flip = True,
	    fill_mode = "nearest")

	augmented_dataset_test = augment_dataset(
	    rescale= 2.0 / 255.0, rotation_range = 10, zoom_range = 0.02, width_shift_range = 0.1,
	    height_shift_range = 0.1, shear_range = 0.01, horizontal_flip = True, vertical_flip = True,
	    fill_mode = "nearest")

	augmented_dataset_outlier = augment_dataset(
	    rescale= 2.0 / 255.0, rotation_range = 10, zoom_range = 0.02, width_shift_range = 0.1,
	    height_shift_range = 0.1, shear_range = 0.01, horizontal_flip = True, vertical_flip = True,
	    fill_mode = "nearest")

	train_path_list = traindf.values
	validate_path_list = valdf.values
	test_path_list = testdf.values
	outlier_path_list = outlierdf.values

	train_img_files = load_all_images(train_path_list, image_height, image_width)
	augmented_dataset_train.fit(train_img_files)
	validate_img_files = load_all_images(validate_path_list, image_height, image_width)
	augmented_dataset_validate.fit(validate_img_files)
	test_img_files = load_all_images(test_path_list, image_height, image_width)
	augmented_dataset_test.fit(test_img_files)
	outlier_img_files = load_all_images(outlier_path_list, image_height, image_width)
	augmented_dataset_outlier.fit(outlier_img_files)

	# Train Images set loading
	train_step_size, train_image_frame = load_dataframe(
	    dataframe = traindf,
	    augmented_props = augmented_dataset_train,
	    directory = None,
	    x_col = "image_path",
	    y_col = None,
	    class_mode="input",
	    target_height = image_height,
	    target_width = image_width,
	    batch_size = BATCH_SIZE,
	    color_mode = "rgb",
	    shuffle = True)

	# Plot samples of Training images
	print("\n================================================================================")
	print("Training Images:")
	print("================================================================================\n")

	per_row = 3
	for i in range(per_row):
	    base = (i * per_row)
	    top = base + per_row
	    train_augmented_images = [train_image_frame[0][0][j] for j in range(base, top)]
	    plotImages(train_augmented_images, 1, per_row, per_row * per_row)

	# Validation Images set loading
	validate_step_size, validate_image_frame = load_dataframe(
	    dataframe = valdf,
	    augmented_props = augmented_dataset_validate,
	    directory = None,
	    x_col = "image_path",
	    y_col = None,
	    class_mode="input",
	    target_height = image_height,
	    target_width = image_width,
	    batch_size = BATCH_SIZE,
	    color_mode = "rgb",
	    shuffle = True)

	# Plot samples Training images
	print("\n================================================================================")
	print("[INFO] Validation Images:")
	print("================================================================================\n")

	per_row = 3
	for i in range(per_row):
	    base = (i * per_row)
	    top = base + per_row
	    validate_augmented_images = [validate_image_frame[0][0][j] for j in range(base, top)]
	    plotImages(validate_augmented_images, 1, per_row, per_row * per_row)

	# Testing Images set loading
	test_step_size, test_image_frame = load_dataframe(
	    dataframe = testdf,
	    augmented_props = augmented_dataset_test,
	    directory = None,
	    x_col = "image_path",
	    y_col = None,
	    class_mode="input",
	    target_height = image_height,
	    target_width = image_width,
	    batch_size = BATCH_SIZE,
	    color_mode = "rgb",
	    shuffle = True)

	# Plot samples Training images
	print("\n================================================================================")
	print("[INFO] Testing Images:")
	print("================================================================================\n")

	per_row = 3
	for i in range(per_row):
	    base = (i * per_row)
	    top = base + per_row
	    test_augmented_images = [test_image_frame[0][0][j] for j in range(base, top)]
	    plotImages(test_augmented_images, 1, per_row, per_row * per_row)

	# Outlier Images set loading
	outlier_step_size, outlier_image_frame = load_dataframe(
	    dataframe = outlierdf,
	    augmented_props = augmented_dataset_outlier,
	    directory = None,
	    x_col = "image_path",
	    y_col = None,
	    class_mode="input",
	    target_height = image_height,
	    target_width = image_width,
	    batch_size = BATCH_SIZE,
	    color_mode = "rgb",
	    shuffle = True)

	# Plot samples Outlier images
	print("\n================================================================================")
	print("[INFO] Outlier Images:")
	print("================================================================================\n")

	per_row = 3
	for i in range(per_row):
	    base = (i * per_row)
	    top = base + per_row
	    outlier_augmented_images = [outlier_image_frame[0][0][j] for j in range(base, top)]
	    plotImages(outlier_augmented_images, 1, per_row, per_row * per_row)
	    
	return (train_image_frame, validate_image_frame, test_image_frame, outlier_image_frame, train_step_size, validate_step_size, test_step_size, outlier_step_size)


def main():

	parse_cli()
	check_dirs()

	print("\n================================================================================")
	print("Num of Epochs: " + str(N_EPOCHS))
	print("Batch Size: " + str(BATCH_SIZE))
	print("Initial Learning Rate: " + str(INIT_LEARNING_RATE))
	print("================================================================================\n")

	(train_image_frame, validate_image_frame, test_image_frame, outlier_image_frame, train_step_size, validate_step_size, test_step_size, outlier_step_size) = build_dataset()

	print("\n================================================================================")
	print("Train Step Size: " + str(train_step_size))
	print("Validate Step Size: " + str(validate_step_size))
	print("Test Step Size: " + str(test_step_size))
	print("Outlier Step Size: " + str(outlier_step_size))
	print("================================================================================\n")

	conv_ae = ConvAutoencoder("conv_autoencoder")
	conv_ae.gen_model(train_image_frame, 
	          validate_image_frame, 
	          test_image_frame,
	          outlier_image_frame,
	          train_step_size, 
	          validate_step_size, 
	          test_step_size,
	          outlier_step_size,
	          image_height, 
	          image_width, 
	          image_depth, 
	          N_EPOCHS,
	          BATCH_SIZE, 
	          INIT_LEARNING_RATE)


if __name__ == "__main__":
	main()
