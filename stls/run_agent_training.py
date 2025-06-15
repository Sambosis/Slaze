# run_agent_training.py

import sys
import os
import runpy
from pathlib import Path

# Get the absolute path of the script's directory, which is the project root
project_root = Path(__file__).parent.resolve()
sys.path.insert(0, str(project_root))

# --- Step 1: Create the Initial Generator Model ---
print("--- Running Initial Generator Creation ---")
generator_creation_script_path = project_root / "create_initial_generator.py"

# Temporarily switch sys.argv for the generator creation script
original_argv = list(sys.argv)
sys.argv = [str(generator_creation_script_path)]

try:
    # Use run_path with an absolute path to avoid ambiguity
    runpy.run_path(str(generator_creation_script_path), run_name="__main__")
    print("--- Initial Generator Creation Successful ---")
except Exception as e:
    print(f"--- Initial Generator Creation Failed: {e} ---")
    sys.exit(1)

# Restore original argv
sys.argv = original_argv

# --- Step 1b: Ensure motif assignments exist ---
motif_assignments_path = project_root / "artifacts" / "motifs" / "motif_assignments.pt"
if not motif_assignments_path.exists():
    print("--- Motif assignments not found. Running motif discovery ---")
    motif_data_dir = project_root / "data" / "processed"
    motif_output_dir = project_root / "artifacts" / "motifs"

    sys.argv = [
        "lspo_3d/train_motifs.py",
        "--data_dir",
        str(motif_data_dir),
        "--output_dir",
        str(motif_output_dir),
        "--num_motifs",
        "2",
        "--model_name",
        "distilbert-base-uncased",
    ]
    try:
        runpy.run_module("lspo_3d.train_motifs", run_name="__main__")
        print("--- Motif discovery completed ---")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"--- Motif discovery failed: {e} ---")
        sys.exit(1)
    finally:
        sys.argv = original_argv

# --- Step 2: Run the Agent Training Script ---
print("\n--- Running Agent Training ---")

# Construct absolute paths for required files and directories
motif_path_abs = project_root / "artifacts" / "motifs" / "motif_assignments.pt"
generator_path_abs = project_root / "lspo_3d" / "initial_generator"
log_dir_abs = project_root / "logs"

# Set the command-line arguments for the training script
sys.argv = [
    "lspo_3d/train_agent.py",
    "--motif-path",
    str(motif_path_abs),
    "--generator-path",
    str(generator_path_abs),
    "--log-dir",
    str(log_dir_abs),
    "--num-motifs",
    "2",
    "--total-timesteps",
    "2048",  # Using a single rollout buffer size for a minimal test
]

# Run the agent training script as a module
runpy.run_module("lspo_3d.train_agent", run_name="__main__")
