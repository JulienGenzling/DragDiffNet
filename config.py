import torch
from dataclasses import dataclass


@dataclass
class Config:
    # General
    seed = 42
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_folds = 1
    n_workers = 10
    num_epochs = 100

    # Data
    data_basepath = "/Data/DrivAerNet"
    cache_dir = "/Data/DrivAerNet/cache"
    train_ratio = 0.8
    augment = False  # Use _flip (flips the mesh aorund one axis) and _aug (scales the mesh by a factor between 1 and 1.2) files
    labels_path = "/Data/DrivAerNet/DrivAerNetPlusPlus_Drag_8k.csv"

    # Model
    inp_feat = "xyz"  # Type of input Features (xyz, HKS, WKS)
    num_eig = 32  # Number of eigenfunctions to use for Spectral Diffusion
    p_in = 3  # Number of input features
    p_out = 1  # Number of output features
    n_block = 4  # Number of DiffusionNetBlock
    n_channels = 64  # Width of the network
    outputs_at = "global_mean"

    # Save dir
    save_dir = "/Data/DrivAerNet/models"
    figures_dir = "/Data/DrivAerNet/figures"

    # Wandb log
    log_wandb = True
