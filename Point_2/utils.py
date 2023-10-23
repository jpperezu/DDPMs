import os
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader


def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()


def save_plot_as_image(images, save_path):
    fig = plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())

    # Save the plot as an image
    plt.axis('off')  # Optionally, turn off axis labels
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close(fig)

def save_plot_as_image_grid(images_list, save_path):
    # Create a grid of images
    num_rows = len(images_list[0])
    num_cols = len(images_list)
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(32, 32))

    for i in range(num_cols):
        for j in range(num_rows):
            axs[j, i].imshow(images_list[i][j].permute(1, 2, 0).cpu())
            axs[j, i].axis('off')

    # Adjust spacing between rows and save the plot as an image
    plt.subplots_adjust(wspace=0, hspace=0)  # Set wspace and hspace to adjust spacing
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close(fig)


def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)


def get_data(args):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(80),  # args.image_size + 1/4 *args.image_size
        torchvision.transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = torchvision.datasets.ImageFolder(args.dataset_path, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader


def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)
