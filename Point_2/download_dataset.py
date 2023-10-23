import kaggle
import os

#LANDSCAPES:
dataset_name = "arnaud58/landscape-pictures"

# Define the folder name where you want to save the images.
download_folder = "landscape_img_folder"

# Create the download folder if it doesn't exist.
os.makedirs(download_folder, exist_ok=True)

# Download the dataset using the Kaggle API and save the files to the specified folder.
kaggle.api.dataset_download_files(dataset_name, path=download_folder, unzip=True)

# CIFAR10:
dataset_name = "joaopauloschuler/cifar10-64x64-resized-via-cai-super-resolution"
download_folder = "cifar10"
os.makedirs(download_folder, exist_ok=True)
kaggle.api.dataset_download_files(dataset_name, path=download_folder, unzip=True)

