import os
import json
import pandas as pd

import torch
from torch.utils.data import Dataset
from utils import get_geometry

from config import Config


class CarMeshDataset(Dataset):
    """Car Mesh Dataset"""

    def __init__(self, fold, train, device):
        """
        fold (int): fold number.
        train (bool): If True, use the training set, else use the test set.
        n_eig (int): Number of eigenvectors to use for processing.
        device (str): device (:D)
        """

        self.train = train
        self.n_eig = Config.num_eig
        self.device = device
        self.basepath = Config.data_basepath
        self.cache_dir = Config.cache_dir
        self.all_items = os.listdir(self.cache_dir)
        self.split_path = os.path.join(self.basepath, "splits.json")

        self.label_map = pd.read_csv(Config.labels_path)

        # Keep meshes that belong to train or test of this fold
        with open(self.split_path, "r") as f:
            split = json.load(f)

        this_fold_files = (
            split[f"fold_{fold}"]["train"] if train else split[f"fold_{fold}"]["test"]
        )
        self.all_items = [item for item in self.all_items if item in this_fold_files]

    def __len__(self):
        return len(self.all_items)

    def __getitem__(self, idx):
        cached_filepath = os.path.join(self.cache_dir, self.all_items[idx])
        verts, faces, frames, mass, L, evals, evecs, gradX, gradY = get_geometry(
            cached_filepath, self.device
        )

        label = self.label_map[self.label_map["Design"] == '_'.join(self.all_items[idx].split("_")[:-1])]["Average Cd"].values[0]

        dict = {
            "vertices": verts,
            "faces": faces,
            "frames": frames,
            "vertex_area": mass,
            "label": torch.tensor(label, dtype=torch.float32),
            "L": L,
            "evals": evals,
            "evecs": evecs,
            "gradX": gradX,
            "gradY": gradY,
        }
        return dict



if __name__ == "__main__":  
    train_dataset = CarMeshDataset(fold=0, train=True, device="cpu")
    print(len(train_dataset))
    print(train_dataset[0])