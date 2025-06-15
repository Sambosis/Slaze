import sys
import os
import runpy

# Get the absolute path of the script's directory, which is the project root
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Construct absolute paths for data and output directories
data_dir_abs = os.path.join(project_root, "data", "processed")
output_dir_abs = os.path.join(project_root, "artifacts", "motifs")

# Set the command-line arguments for the script
sys.argv = [
    "lspo_3d/train_motifs.py",
    "--data_dir",
    data_dir_abs,
    "--output_dir",
    output_dir_abs,
    "--num_motifs",
    "2",
    "--model_name",
    "sentence-transformers/all-MiniLM-L6-v2",
]

# Run the script as a module
runpy.run_module("lspo_3d.train_motifs", run_name="__main__")
