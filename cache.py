import os
from multiprocessing import Pool
from functools import partial
from trimesh import load_mesh
from diffusion_utils import save_operators
from tqdm import tqdm
import torch
import numpy as np
from config import Config


def prepopulate_cache_file(mesh_file_path: str, cache_dir: str, n_eig: int) -> None:
    """
    Pre-populates the cache for 1 mesh.

    Parameters:
    - mesh_file_path (str): Path to the mesh file.
    - cache_dir (str): Path to the cache directory.
    - n_eig (int): Number of eigenvalues to use.
    """

    # Check if this mesh has been processed with this number of eigenvalues
    npz_file = mesh_file_path.split("/")[-1].split(".")[0] + "_" + str(n_eig) + ".npz"
    save_path = os.path.join(cache_dir, npz_file)

    if not os.path.exists(save_path):
        # Load the mesh using trimesh
        mesh = load_mesh(mesh_file_path, process=False)

        # Extract vertices and faces
        verts = mesh.vertices
        faces = mesh.faces

        save_operators(torch.tensor(verts), torch.tensor(faces), n_eig, save_path)


def prepopulate_cache(
    data_basepath: str,
    cache_dir: str,
    n_eig: int,
    n_workers: int,
    use_augmentation: bool,
) -> None:
    """
    Pre-populates the cache directory with all available meshes.

    Parameters:
    - data_basepath (str): Path to the base data directory.
    - cache_dir (str): Path to the cache directory.
    - n_eig (int): Number of eigenvectors to use for processing.
    - n_workers (int): Number of workers to use for data loading.
    """

    os.makedirs(cache_dir, exist_ok=True)

    if use_augmentation:
        all_obj_files = [
            os.path.join(data_basepath, "mesh_aug", f)
            for f in os.listdir(os.path.join(data_basepath, "mesh_aug"))
            if not f.endswith("_aug.obj")
        ]
    else:
        all_obj_files = [
            os.path.join(data_basepath, "mesh_resampled", f)
            for f in os.listdir(os.path.join(data_basepath, "mesh_resampled"))
        ]

    worker = partial(prepopulate_cache_file, cache_dir=cache_dir, n_eig=n_eig)

    with Pool(processes=n_workers) as pool:
        for _ in tqdm(
            pool.imap(worker, all_obj_files),
            total=len(all_obj_files),
            desc="Caching files",
        ):
            pass


if __name__ == "__main__":
    print("Caching data (might take a while)...")
    prepopulate_cache(
        Config.data_basepath,
        cache_dir=Config.cache_dir,
        n_eig=Config.num_eig,
        n_workers=Config.n_workers,
        use_augmentation=Config.augment,
    )
