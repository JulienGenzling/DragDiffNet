import os
import pymeshlab
from tqdm import tqdm

def resample_mesh(input_path, output_path, target_faces=10000):
    """
    Resample a mesh to a target number of faces and save the result.

    Parameters:
    - input_path (str): Path to the input .stl file.
    - output_path (str): Path to save the resampled .stl file.
    - target_faces (int): Number of target faces for the resampled mesh.
    """
    # Load the mesh
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(input_path)

    # Apply mesh simplification
    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=target_faces)

    # Save the resampled mesh
    ms.save_current_mesh(output_path)

def resample_all_meshes(input_dir, output_dir, target_faces=10000):
    """
    Resample all .stl meshes in a directory.

    Parameters:
    - input_dir (str): Path to the directory containing input .stl files.
    - output_dir (str): Path to the directory where resampled .stl files will be saved.
    - target_faces (int): Number of target faces for resampled meshes.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get a list of all .stl files in the input directory
    stl_files = [f for f in os.listdir(input_dir) if f.endswith('.stl')]

    # Process each .stl file
    for stl_file in tqdm(stl_files, desc="Resampling meshes"):
        input_path = os.path.join(input_dir, stl_file)
        output_path = os.path.join(output_dir, stl_file)
        if not os.path.exists(output_path):
            try:
                resample_mesh(input_path, output_path, target_faces=target_faces)
            except:
                continue

# Define paths
input_directory = "/Data/DrivAerNet/mesh"
output_directory = "/Data/DrivAerNet/mesh_resampled"

# Resample all meshes
resample_all_meshes(input_directory, output_directory, target_faces=30000)
