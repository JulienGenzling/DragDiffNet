import os
import json
import pandas as pd

import torch
from torch.utils.data import Dataset
from utils import get_geometry


class CarMeshDataset(Dataset):
    """RNA Mesh Dataset"""

    def __init__(self, data_dir, fold, train, n_eig, device):
        """
        root_dir (string): Directory with all the meshes.
        train (bool): If True, use the training set, else use the test set.
        op_cache_dir (string): Directory to cache the operators.
        n_classes (int): Number of classes.
        device (str): device (:D)
        """

        self.train = train
        self.n_eig = n_eig
        self.device = device
        self.cache_dir = os.path.join(data_dir, "cache")
        self.all_items = os.listdir(self.cache_dir)
        self.split_path = os.path.join(data_dir, "splits.json")

        self.label_map = pd.read_csv(os.path.join(data_dir, "drag_coeffs.csv"))

        # Keep meshes that belong to train or test of this fold
        with open(self.split_path, "r") as f:
            split = json.load(f)

        this_fold_files = (
            split[f"fold_{fold}"]["train"] if train else split[f"fold_{fold}"]["test"]
        )

        self.all_items = [item.split("_")[0] for item in self.all_items if item.split("_")[0] in this_fold_files]

    def __len__(self):
        return len(self.all_items)

    def __getitem__(self, idx):
        cached_filepath = os.path.join(self.cache_dir, f"{self.all_items[idx]}_{self.n_eig}.npz")
        verts, faces, frames, mass, L, evals, evecs, gradX, gradY = get_geometry(
            cached_filepath, self.device
        )
        label = self.label_map[
            self.label_map["file"] == self.all_items[idx]
        ]["Cd"].values[0]
        dict = {
            "vertices": verts,
            "faces": faces,
            "frames": frames,
            "vertex_area": mass,
            "label": torch.tensor(label),
            "L": L,
            "evals": evals,
            "evecs": evecs,
            "gradX": gradX,
            "gradY": gradY,
        }
        return dict
