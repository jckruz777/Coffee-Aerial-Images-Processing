import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

from PIL import Image
from plotly import tools
from sklearn.manifold import TSNE


def plot_val_outlier_scatter(val_dataset, title, img_path, outlier_dataset):
    tsne = TSNE(n_components=2, random_state=0)
    z_tsne_val = tsne.fit_transform(val_dataset)
    z_tsne_outlier = tsne.fit_transform(outlier_dataset)
    plt.style.use("ggplot")
    f = plt.figure()
    fig, ax = plt.subplots()
    colors = ['tab:blue', 'tab:red']
    labels = ['validate', 'outlier']
    ax.scatter(z_tsne_val[:, 0], z_tsne_val[:, 1], c=colors[0], label=labels[0])
    ax.scatter(z_tsne_outlier[:, 0], z_tsne_outlier[:, 1], c=colors[1], label=labels[1])
    ax.legend()
    ax.grid(True)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.title(title + ' Data Space')
    #plt.show()
    f.savefig(img_path, bbox_inches='tight')


def plot_latent_space(model, dataset, title, img_path, dataset_2=None, batch_size=32):
    # Visualization of latent space
    tsne = TSNE(n_components=2, random_state=0)
    z_mean = model.predict(dataset)
    z_tsne = tsne.fit_transform(z_mean)
    plt.style.use("ggplot")
    f = plt.figure()
    if dataset_2:
        z_mean_outlier = model.predict(dataset_2)
        z_tsne_outlier = tsne.fit_transform(z_mean_outlier)
        fig, ax = plt.subplots()
        colors = ['tab:blue', 'tab:red']
        labels = ['validate', 'outlier']
        ax.scatter(z_tsne[:, 0], z_tsne[:, 1], c=colors[0], label=labels[0])
        ax.scatter(z_tsne_outlier[:, 0], z_tsne_outlier[:, 1], c=colors[1], label=labels[1])
        ax.legend()
        ax.grid(True)
    else:
        plt.scatter(z_tsne[:, 0], z_tsne[:, 1])
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.title(title + ' Data Latent Space')
    #plt.show()
    f.savefig(img_path, bbox_inches='tight')


def plotImages(images_arr, rows, cols, size):
    fig, axes = plt.subplots(rows, cols, figsize=(size,size))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        print("IMG MIN = " + str(np.min(img)))
        print("IMG MAX = " + str(np.max(img)))
        ax.imshow((img * int(255 / 2)).astype("uint8") + int(255 / 2))
    plt.tight_layout()
    #plt.show()


def visualize_predictions(decoded, gt, samples=5):
    # initialize our list of output images
    outputs = None

    # loop over our number of output samples
    for i in range(samples):
        # grab the original image and reconstructed image
        original = (gt[0][0][i] * int(255 / 2)).astype("uint8") + int(255 / 2)
        recon = (decoded[i] * int(255 / 2)).astype("uint8") + int(255 / 2)

        # stack the original and reconstructed image side-by-side
        output = np.hstack([original, recon])

        # if the outputs array is empty, initialize it as the current
        # side-by-side image display
        if outputs is None:
            outputs = output

        # otherwise, vertically stack the outputs
        else:
            outputs = np.vstack([outputs, output])

    # return the output images
    return outputs


def plot_metrics(n_epochs, 
                 out_plot_file_loss,
                 out_plot_file_acc,
                 model_H):
    N = np.arange(0, n_epochs)
    # construct a plot that plots and saves the training history loss
    plt.style.use("ggplot")
    f = plt.figure()
    plt.plot(N, model_H.history["loss"], label="train_loss")
    plt.plot(N, model_H.history["val_loss"], label="val_loss")
    #plt.plot(N, loss_array, label="train_loss")
    #plt.plot(N, val_loss_array["val_loss"], label="val_loss")
    plt.title("Training Loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")
    #plt.show()
    f.savefig(out_plot_file_loss, bbox_inches='tight')

    # construct a plot that plots and saves the training history accuracy
    plt.style.use("ggplot")
    f = plt.figure()
    plt.plot(N, model_H.history["acc"], label="train_acc")
    plt.plot(N, model_H.history["val_acc"], label="val_acc")
    #plt.plot(N, loss_array, label="train_loss")
    #plt.plot(N, val_loss_array["val_loss"], label="val_loss")
    plt.title("Training Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower left")
    #plt.show()
    f.savefig(out_plot_file_acc, bbox_inches='tight')


def make_predictions(autoencoder,
                    testX,
                    trainX,
                    outlierX, 
                    recons_visual_file_test,
                    recons_visual_file_train,
                    recons_visual_file_outlier):
    # use the convolutional autoencoder to make predictions on the
    # testing images, construct the visualization, and then save it
    # to disk
    print("[INFO] Making predictions on TESTING images...")
    decoded = autoencoder.predict(testX[0][0])
    vis = visualize_predictions(decoded, testX)
    cv2.imwrite(recons_visual_file_test, vis)
    #cv2.imshow(vis)

    # use the convolutional autoencoder to make predictions on the
    # training images, construct the visualization
    print("[INFO] Making predictions on TRAINING images...")
    decoded_train = autoencoder.predict(trainX[0][0])
    vis_train = visualize_predictions(decoded_train, trainX)
    cv2.imwrite(recons_visual_file_train, vis_train)
    #cv2.imshow(vis_train)

    # use the convolutional autoencoder to make predictions on the
    # outlier images, construct the visualization
    print("[INFO] Making predictions on OUTLIER images...")
    decoded_outlier = autoencoder.predict(outlierX[0][0])
    vis_outlier = visualize_predictions(decoded_outlier, outlierX)
    cv2.imwrite(recons_visual_file_outlier, vis_outlier)
    #cv2.imshow(vis_outlier)
