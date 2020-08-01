#############
# Constants #
#############

EPSILON = 1e-8

# Image Features

image_width = 100 - 4
image_height = 100 - 4
image_depth = 3
extension = ".JPG"
original_dim = image_width * image_height
anomalyTreshold = 0.45

#totalTrain = 103680
#totalVal = 9728
#totalTest = 9728

#train_percent = 42
#val_percent = 4
#test_percent = 4

train_percent = 1.5
val_percent = 1
test_percent = 1

# Dropout Percentage
dropout_percent = 0.05

# Paths
coffee_dataset_path = "../../../../coffee_dataset/ICAFÉ_proyecto/"
models_path = "../Coffee_Project/AE_Msc/model_checkpoints/"
tensorboard_path = "../Coffee_Project/AE_Msc/TensorBoard_logs/"
csv_path = "../Coffee_Project/AE_Msc/csv_dataset/"
res_path = "../Coffee_Project/AE_Msc/trained_results/"

# .CSV file names
train_csv_name = csv_path + "train_data.csv"
val_csv_name = csv_path + "val_data.csv"
test_csv_name = csv_path + "test_data.csv"
outlier_csv_name = csv_path + "outlier_data.csv"

# Results Paths

# path to the scatter for validate and outlier datasets
plot_scatter_file_valoutlier = res_path + "plot_scatter_valoutlier.pdf"
# path to output dataset file
out_dataset = res_path + "images.pickle"
# path to output trained autoencoder
model_path = res_path + "autoencoder.model"
# path to output reconstruction visualization file for testing
recons_visual_file_test = res_path + "recon_vis_test.png"
# path to output reconstruction visualization file for training
recons_visual_file_train = res_path + "recon_vis_train.png"
# path to output reconstruction visualization file for outlier
recons_visual_file_outlier = res_path + "recon_vis_outlier.png"

# path to output plot files
out_plot_file_loss = res_path + "plot_loss.pdf"
out_plot_file_acc = res_path + "plot_acc.pdf"
out_plot_file_latent_train = res_path + "plot_latent_train.pdf"
out_plot_file_latent_test = res_path + "plot_latent_test.pdf"
out_plot_file_latent_val = res_path + "plot_latent_val.pdf"
out_plot_file_latent_valoutlier = res_path + "plot_latent_valoutlier.pdf"
