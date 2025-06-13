import argparse
import os
from pathlib import Path
from typing import List

import torch

# Internal project imports
from 3d_lspo.data.processor import process_raw_models
from 3d_lspo.models.motif_encoder import MotifEncoder


def setup_arguments() -> argparse.ArgumentParser:
    """
    Configures and returns the argument parser for command-line execution.
    """
    parser = argparse.ArgumentParser(
        description="Train Motif Encoder and Discover Design Motifs."
    )

    # --- Path Arguments ---
    parser.add_argument(
        "--raw_data_dir",
        type=str,
        required=True,
        help="Path to the directory containing raw 3D model files (.stl, .step)."
    )
    parser.add_argument(
        "--processed_data_dir",
        type=str,
        required=True,
        help="Path to save/load the processed CSG design traces (.txt files)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./artifacts/motifs",
        help="Directory to save the trained model artifacts."
    )

    # --- Model Hyperparameters ---
    parser.add_argument(
        "--model_name",
        type=str,
        default='bert-base-uncased',
        help="Name of the pretrained model from Hugging Face hub (e.g., 'bert-base-uncased')."
    )
    parser.add_argument(
        "--n_motifs",
        type=int,
        default=100,
        help="Number of motifs to discover (the 'k' in k-means)."
    )

    # --- Training Parameters ---
    parser.add_argument(
        "--force_reprocess",
        action='store_true',
        help="If set, forces reprocessing of raw data even if processed files exist."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training ('cuda' or 'cpu')."
    )

    return parser


def load_or_process_data(raw_dir: str, processed_dir: str, force_reprocess: bool) -> List[str]:
    """
    Loads processed design traces or generates them if they don't exist.
    """
    processed_path = Path(processed_dir)
    raw_path = Path(raw_dir)

    # Check if we need to run the processing step
    should_process = force_reprocess or not processed_path.exists() or not any(processed_path.iterdir())

    if should_process:
        print(f"Processing raw data from '{raw_path}' into '{processed_path}'.")
        if not raw_path.exists():
            print(f"Error: Raw data directory not found at '{raw_path}'.")
            return []
        try:
            # Assuming process_raw_models creates the output dir
            process_raw_models(raw_path, processed_path)
            print("Raw data processing complete.")
        except Exception as e:
            print(f"An error occurred during data processing: {e}")
            return []

    # Load the processed text files
    design_traces = []
    if not processed_path.exists():
        print(f"Error: Processed data directory not found at '{processed_path}' after processing attempt.")
        return []

    print(f"Loading design traces from '{processed_path}'...")
    for trace_file in processed_path.rglob('*.txt'):
        try:
            design_traces.append(trace_file.read_text())
        except Exception as e:
            print(f"Could not read file {trace_file}: {e}")
    return design_traces


def main(args: argparse.Namespace) -> None:
    """
    The main execution function for the motif discovery training script.
    """
    print("--- Starting Phase 1: Motif Discovery ---")
    print(f"Configuration: {args}")

    # 1. Load or process the design traces
    design_traces = load_or_process_data(
        raw_dir=args.raw_data_dir,
        processed_dir=args.processed_data_dir,
        force_reprocess=args.force_reprocess
    )
    if not design_traces:
        print("No design traces found or generated. Exiting.")
        return
    print(f"Loaded {len(design_traces)} design traces.")

    # 2. Initialize the MotifEncoder
    print(f"Initializing MotifEncoder with model '{args.model_name}' on device '{args.device}'...")
    motif_encoder = MotifEncoder(
        model_name=args.model_name,
        num_motifs=args.n_motifs
    )
    print("MotifEncoder initialized.")

    # 3. Generate embeddings for all traces
    print("Generating embeddings for all design traces...")
    try:
        embeddings = motif_encoder.encode(design_traces).cpu().numpy()
        print(f"Generated {embeddings.shape[0]} embeddings with dimension {embeddings.shape[1]}.")
    except Exception as e:
        print(f"Failed to generate embeddings: {e}")
        return

    # 4. Perform clustering to find motifs
    print(f"Performing k-means clustering for {args.n_motifs} motifs...")
    try:
        motif_encoder.discover_motifs(embeddings)
        print("Motif discovery complete.")
    except Exception as e:
        print(f"Failed to perform clustering: {e}")
        return

    # 5. Save the trained model and motifs
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving trained encoder and k-means model to {output_path}...")
    try:
        motif_encoder.save(output_path)
        print(f"Artifacts saved successfully.")
    except Exception as e:
        print(f"Failed to save model artifacts: {e}")
        return

    print("--- Motif Discovery Phase finished successfully ---")


if __name__ == "__main__":
    parser = setup_arguments()
    args = parser.parse_args()
    main(args)
