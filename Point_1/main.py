# -------------- Adapted from https://huggingface.co/blog/annotated-diffusion --------------# 

# Step-by-step PyTorch implementation --> based on Phil Wang's implementation --> which itself is based on the original TensorFlow implementation.

# Note that there are several perspectives on diffusion models.
# Here, we employ the discrete-time (latent variable model) perspective

# Assuming you have PyTorch installed: Require: pip install -q -U einops datasets matplotlib tqdm

import math
from inspect import isfunction
from functools import partial
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
import torch
from torch import nn, einsum
import torch.nn.functional as F
from PIL import Image
import requests
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from pathlib import Path
from torch.optim import Adam
from torchvision.utils import save_image
import matplotlib.animation as animation
import argparse
import os

# ----------------------------------------
# Parser:
# ----------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--schedule', type=str, choices=['cosine', 'linear', 'quadratic', 'sigmoid'], default='linear', help='Select the schedule function')
parser.add_argument('--timesteps', type=int, default=300, help='Number of timesteps')
parser.add_argument('--loss_type', type=str, choices=['l1', 'l2', 'huber'], default='huber', help='Loss function type')
parser.add_argument('--epochs', type=int, default=6, help='Number of training epochs')
args = parser.parse_args()


# ----------------------------------------
# Newtork helpers:
# ----------------------------------------

# First, we define some helper functions and classes which will be used when implementing
# the neural network. Importantly, we define a Residual module, which simply adds the input
# to the output of a particular function (in other words, adds a residual connection to a 
# particular function). We also define aliases for the up- and downsampling operations.

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1),
    )


def Downsample(dim, dim_out=None):
    # No More Strided Convolutions or Pooling
    return nn.Sequential(
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1),
    )

# ----------------------------------------
# Position embeddings:
# ----------------------------------------

# As the parameters of the neural network are shared across time (noise level), the
# authors employ sinusoidal position embeddings to encode t, inspired by the Transformer  
# (Vaswani et al., 2017).This makes the neural network "know" at which particular time step
# (noise level) it is operating, for every image in a batch.

# The SinusoidalPositionEmbeddings module takes a tensor of shape (batch_size, 1) as input
# (i.e. the noise levels of several noisy images in a batch), and turns this into a tensor
# of shape (batch_size, dim), with dim being the dimensionality of the position embeddings.
# This is then added to each residual block, as we will see further.

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

# ----------------------------------------
# ResNet block
# ----------------------------------------

# Next, we define the core building block of the U-Net model. 
# The DDPM authors employed a Wide ResNet block, but Phil Wang has replaced the standard
# convolutional layer by a "weight standardized" version, which works better in combination
# with group normalization 

class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, "o ... -> o 1 1 1", "mean")
        var = reduce(weight, "o ... -> o 1 1 1", partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    """https://arxiv.org/abs/1512.03385"""

    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
            if exists(time_emb_dim)
            else None
        )

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)
    
# ----------------------------------------
# Attention module
# ----------------------------------------

# Next, we define the attention module, which the DDPM authors added in between the convolutional
# blocks. Phil Wang employs 2 variants of attention: one is regular multi-head self-attention
# (as used in the Transformer), the other one is a linear attention variant (Shen et al., 2018),
# whose time- and memory requirements scale linear in the sequence length, as opposed to quadratic
# for regular attention.

class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), 
                                    nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)

# ----------------------------------------
# Group normalization
# ----------------------------------------

# The DDPM authors interleave the convolutional/attention layers of the U-Net with group
# normalization (Wu et al., 2018). Below, we define a PreNorm class, which will be used 
# to apply groupnorm before the attention layer, as we'll see further. Note that there's
# been a debate about whether to apply normalization before or after attention in Transformers.

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)
    
# ----------------------------------------
# Conditional U-Net
# ----------------------------------------

# Now that we've defined all building blocks (position embeddings, ResNet blocks, attention
# and group normalization), it's time to define the entire neural network. 
# Recall that the job of the network εθ(xt,t) is to take in a batch of noisy images and 
# their respective noise levels, and output the noise added to the input. More formally:

    # the network takes a batch of noisy images of shape (batch_size, num_channels, height, width)
    # and a batch of noise levels of shape (batch_size, 1) as input, and returns a tensor of shape
    # (batch_size, num_channels, height, width)

# The network is built up as follows:

    # (1) First, a convolutional layer is applied on the batch of noisy images, and position embeddings
    # are computed for the noise levels

    # (2) Next, a sequence of downsampling stages are applied. Each downsampling stage consists of
    # 2 ResNet blocks + groupnorm + attention + residual connection + a downsample operation

    # (3) At the middle of the network, again ResNet blocks are applied, interleaved with attention

    # (4) Next, a sequence of upsampling stages are applied. Each upsampling stage consists of 2
    # ResNet blocks + groupnorm + attention + residual connection + an upsample operation

    # (5) Finally, a ResNet block followed by a convolutional layer is applied.

# Ultimately, neural networks stack up layers as if they were lego blocks:

class Unet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        self_condition=False,
        resnet_block_groups=4,
    ):
        super().__init__()

        # determine dimensions
        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 1, padding=0) # changed to 1 and 0 from 7,3

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings
        time_dim = dim * 4

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Downsample(dim_in, dim_out)
                        if not is_last
                        else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Upsample(dim_out, dim_in)
                        if not is_last
                        else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )

        self.out_dim = default(out_dim, channels)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, x, time, x_self_cond=None):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)

 
# ----------------------------------------
# Defining the forward diffusion process
# ----------------------------------------

    # Define various schedules for the T timesteps:

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = None #TODO Define betas
    return torch.clip(betas, 0.0001, 0.9999)

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


    # To start with, let's use the linear schedule for T = 300 time steps and define
    # the various variables from the βt which we will need, such as the cumulative product
    # of the variances αˉt.Each of the variables below are just 1-dimensional tensors, storing
    # values from t to T. 
    # Importantly, we also define an extract function, which will allow us to extract the appropriate 
    # t index for a batch of indices.

# Select beta schedule
def select_schedule(schedule_type, timesteps):
    if schedule_type == 'cosine':
        return cosine_beta_schedule(timesteps)
    elif schedule_type == 'linear':
        return linear_beta_schedule(timesteps)
    elif schedule_type == 'quadratic':
        return quadratic_beta_schedule(timesteps)
    elif schedule_type == 'sigmoid':
        return sigmoid_beta_schedule(timesteps)

betas = select_schedule(args.schedule, args.timesteps)

# define alphas 
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

# calculations for diffusion q(x_t | x_{t-1}) and others
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# calculations for posterior 
posterior_variance = None #TODO Define posterior q(x_{t-1} | x_t, x_0) 

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


    # We'll illustrate with an example image how noise is added at each time step of the diffusion process:

url = 'https://www.nme.com/wp-content/uploads/2017/11/NF-GettyImages-613730118.jpg'
image = Image.open(requests.get(url, stream=True).raw) 

# Noise is added to PyTorch tensors, rather than Pillow Images. We'll first define image transformations
# that allow us to go from a PIL image to a PyTorch tensor (on which we can add the noise), and vice versa.

# These transformations are fairly simple: we first normalize images by dividing by 255 (such that they
# are in the [0,1] range), and then make sure they are in the [−1,1] range. From the DPPM paper:

    # "We assume that image data consists of integers in {0,1,...,255} scaled linearly to 
    # [−1,1]. This ensures that the neural network reverse process operates on consistently
    # scaled inputs starting from the standard normal prior p(x_{T})

image_size = 128
transform = Compose([
    Resize(image_size),
    CenterCrop(image_size),
    ToTensor(), # turn into torch Tensor of shape CHW, divide by 255
    Lambda(lambda t: (t * 2) - 1),
    
])

x_start = transform(image).unsqueeze(0)

# We also define the reverse transform, which takes in a PyTorch tensor containing values in 
# [−1,1] and turn them back into a PIL image:

reverse_transform = Compose([
     Lambda(lambda t: (t + 1) / 2),
     Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
     Lambda(lambda t: t * 255.),
     Lambda(lambda t: t.numpy().astype(np.uint8)),
     ToPILImage(),
])

    # We can now define the forward diffusion process as in the paper:

# forward diffusion (using the cumproduct of ̅α)
def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = None #TODO Define noise

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )

    return None #TODO Define output for a noiser image in the forward diffusion

# If you want to test it on a particular time step:

def get_noisy_image(x_start, t):
  # add noise
  x_noisy = q_sample(x_start, t=t)

  # turn back into PIL image
  noisy_image = reverse_transform(x_noisy.squeeze())

  return noisy_image

#Let's visualize this for various time steps:

# use seed for reproducibility
torch.manual_seed(0)

# source: https://pytorch.org/vision/stable/auto_examples/plot_transforms.html#sphx-glr-auto-examples-plot-transforms-py
def plot(imgs, with_orig=False, row_title=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0]) + with_orig
    fig, axs = plt.subplots(figsize=(200,200), nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        row = [image] + row if with_orig else row
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if with_orig:
        axs[0, 0].set(title='Original image')
        axs[0, 0].title.set_size(8)
    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()

plot([get_noisy_image(x_start, torch.tensor([t])) for t in np.floor(np.linspace(0, args.timesteps-1, 15)).astype(int)])

plt.savefig('Forward.png') #Change according to your preference 


# Beta vs timestep plot

timesteps=args.timesteps
time_steps = list(range(timesteps))

# Calculate betas for each schedule
betas_cosine = cosine_beta_schedule(args.timesteps)
betas_linear = linear_beta_schedule(args.timesteps)
betas_quadratic = quadratic_beta_schedule(args.timesteps)
betas_sigmoid = sigmoid_beta_schedule(args.timesteps)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(time_steps, betas_cosine, label='Cosine', linewidth=2)
plt.plot(time_steps, betas_linear, label='Linear', linewidth=2)
plt.plot(time_steps, betas_quadratic, label='Quadratic', linewidth=2)
plt.plot(time_steps, betas_sigmoid, label='Sigmoid', linewidth=2)

plt.xlabel('Time Step')
plt.ylabel('Betas')
plt.title('Betas vs. Time Step for Different Schedules')
plt.legend()
plt.grid(True)
plt.savefig('betas_plot.png')

# Calculate sqrt(1 - betas) for each schedule
sqrt_one_minus_betas_cosine = torch.sqrt(1 - betas_cosine)
sqrt_one_minus_betas_linear = torch.sqrt(1 - betas_linear)
sqrt_one_minus_betas_quadratic = torch.sqrt(1 - betas_quadratic)
sqrt_one_minus_betas_sigmoid = torch.sqrt(1 - betas_sigmoid)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(time_steps, sqrt_one_minus_betas_cosine, label='Cosine', linewidth=2)
plt.plot(time_steps, sqrt_one_minus_betas_linear, label='Linear', linewidth=2)
plt.plot(time_steps, sqrt_one_minus_betas_quadratic, label='Quadratic', linewidth=2)
plt.plot(time_steps, sqrt_one_minus_betas_sigmoid, label='Sigmoid', linewidth=2)

plt.xlabel('Time Step')
plt.ylabel('sqrt(1 - Betas)')
plt.title('sqrt(1 - Betas) vs. Time Step for Different Schedules')
plt.legend()
plt.grid(True)
plt.savefig('sqrt_one_minus_betas_plot.png')


    # So, we can now define the loss function given the model as follows:
    # Define L1, L2 and Huber loss to choose:

def p_losses(denoise_model, x_start, t, noise=None, loss_type="l1"):
    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy = q_sample(x_start=x_start, t=t, noise=noise)
    predicted_noise = denoise_model(x_noisy, t)

    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss

# The denoise_model will be our U-Net defined above. We'll employ the Huber loss between
# the true and the predicted noise.


# ----------------------------------------
# Define a PyTorch Dataset + DataLoader
# ----------------------------------------

# Here we define a regular PyTorch Dataset. The dataset simply consists of images from
# a real dataset, like Fashion-MNIST, CIFAR-10 or ImageNet, scaled linearly to [−1,1].

# Each image is resized to the same size. Interesting to note is that images are also
# randomly horizontally flipped. From the paper:

    # "We used random horizontal flips during training for CIFAR10; we tried training both with
    #  and without flips, and found flips to improve sample quality slightly"

# Here we use the 🤗 Datasets library to easily load the Fashion MNIST dataset from the hub. 
# This dataset consists of images which already have the same resolution, namely 28x28.

# load dataset from the hub
dataset = load_dataset("fashion_mnist")
image_size = 28
channels = 1
batch_size = 128

# Next, we define a function which we'll apply on-the-fly on the entire dataset. 
# We use the with_transform functionality for that. The function just applies some
# basic image preprocessing: random horizontal flips, rescaling and finally make 
# them have values in the [−1,1] range:

# define image transformations (e.g. using torchvision)
transform = Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)
])

# define function
def transforms(examples):
   examples["pixel_values"] = [transform(image.convert("L")) for image in examples["image"]]
   del examples["image"]

   return examples

transformed_dataset = dataset.with_transform(transforms).remove_columns("label")

# create dataloader
dataloader = DataLoader(transformed_dataset["train"], batch_size=batch_size, shuffle=True)

batch = next(iter(dataloader))



# ----------------------------------------
# Sampling
# ----------------------------------------

# As we'll sample from the model during training (in order to track progress),
# we define the code for that below. Sampling is summarized in the Denoising Diffusion
# Probabilistic Models paper as Algorithm 2:

@torch.no_grad()
def p_sample(model, x, t, t_index):
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)
    
    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    model_mean = None #TODO 

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return None #TODO

# Algorithm 2 (including returning all images)
@torch.no_grad()

def p_sample_loop(model, shape):
    device = next(model.parameters()).device

    b = shape[0]
    # start from pure noise (for each example in the batch)
    img = torch.randn(shape, device=device)
    imgs = []

    for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
        img = p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i)
        imgs.append(img.cpu().numpy())
    return imgs

@torch.no_grad()
def sample(model, image_size, batch_size=16, channels=3):
    return p_sample_loop(model, shape=(batch_size, channels, image_size, image_size))

# Note that the code above is a simplified version of the original implementation. 
# We found our simplification (which is in line with Algorithm 2 in the paper) to
# work just as well as the original, more complex implementation, which employs clipping.
# See: https://github.com/hojonathanho/diffusion/issues/5



# ----------------------------------------
# Training
# ----------------------------------------

# Next, we train the model in regular PyTorch fashion. 

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

#Define folder to store trained models:
folder_models = "models" #Change according to your preference
if not os.path.exists(folder_models):
    os.makedirs(folder_models)
model_path = os.path.join(folder_models, 'trained_model.pt') #Change according to your preference

#Below, we define the model, and move it to the GPU. We also define a standard optimizer (Adam).

device = "cuda" if torch.cuda.is_available() else "cpu"

model = Unet(
    dim=image_size,
    channels=channels,
    dim_mults=(1, 2, 4,)
)
model.to(device)

optimizer = Adam(model.parameters(), lr=1e-3)

#Let's start training!

epochs = args.epochs

for epoch in tqdm(range(epochs), desc='Training...'):
    for step, batch in tqdm(enumerate(dataloader), desc='Loading batch...'):
        optimizer.zero_grad()

        batch_size = batch["pixel_values"].shape[0]
        batch = batch["pixel_values"].to(device)

        # Algorithm 1 line 3: sample t uniformally for every example in the batch
        t = torch.randint(0, timesteps, (batch_size,), device=device).long()

        loss = p_losses(model, batch, t, loss_type=args.loss_type)

        if step % 100 == 0:
            print("Loss:", loss.item())

        loss.backward()
        optimizer.step()
    
        # Save model
        torch.save(model.state_dict(), model_path)



# ----------------------------------------
# Sampling (inference)
# ----------------------------------------

# sample 64 images
model.load_state_dict(torch.load(model_path))
model.eval()
samples = sample(model, image_size=image_size, batch_size=64, channels=channels)

# show a random one
# random_index = 5
# fig, ax = plt.subplots()
# # Display the image on the axis
# ax.imshow(samples[-1][random_index].reshape(image_size, image_size, channels), cmap="gray")
# plt.savefig("sample.png")

# Show 9 random images
num_samples = 9
fig, axes = plt.subplots(3, 3, figsize=(10, 10))
for i in range(num_samples):
    row = i // 3
    col = i % 3
    ax = axes[row, col]
    ax.imshow(samples[-1][i].reshape(image_size, image_size, channels), cmap="gray")
    ax.axis('off')
plt.savefig("sample_subplot.png")

# Keep in mind that the dataset we trained on is pretty low-resolution (28x28)

# We can also create a gif of the denoising process for a random sample:

random_index = 53

fig = plt.figure()
ims = []
for i in range(timesteps):
    im = plt.imshow(samples[i][random_index].reshape(image_size, image_size, channels), cmap="gray", animated=True)
    ims.append([im])

animate = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
animate.save('diffusion.gif')

    