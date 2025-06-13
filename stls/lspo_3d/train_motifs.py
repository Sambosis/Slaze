# -*- coding: utf-8 -*-
"""Main script for Phase 1: Training the MotifEncoder and discovering design motifs.

This script orchestrates the process of learning a latent space of 3D design
motifs from a corpus of existing 3D models. It performs the following steps:
1.  Loads design traces (sequences of CSG operations) which are assumed to be
    pre-processed into text files.
2.  Initializes a Transformer-based encoder (`MotifEncoder`) using a pre-trained
    model from Hugging Face. No fine-tuning is performed on the encoder.
3.  Uses the encoder to generate embeddings for the entire dataset of design traces.
4.  Performs k-means clustering on the embeddings to identify a set of
    high-level design motifs (cluster centroids).
5.  Saves the trained k-means model (as part of the MotifEncoder), the discovered
    motif centroids, and cluster assignments to disk for use in downstream tasks.

This script is intended to be run from the command line.
Example:
    python -m lspo_3d.train_motifs --data_dir ./data/processed_traces \
                                  --output_dir ./artifacts/motifs \
                                  --model_name bert-base-uncased \
                                  --num_motifs 50
"""

import argparse
import os
import logging
from pathlib import Path
from typing import List

import torch
import numpy as np
from tqdm import tqdm

# Internal project imports
# Assuming that processor.py has been run and has outputted text files with CSG traces.
# from src.lspo_3d.data.processor import process_raw_models # Not used; we load pre-processed traces.
from lspo_3d.models.motif_encoder import MotifEncoder

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def setup_arg_parser() -> argparse.ArgumentParser:
    """Sets up the command-line argument parser.

    Defines arguments for data paths, model hyperparameters, and output
    directories.

    Returns:
        argparse.ArgumentParser: The configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Discover design motifs from CSG traces using a Transformer encoder and K-Means clustering.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the directory containing pre-processed design trace files (.txt).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the MotifEncoder, k-means model, and motif artifacts.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="bert-base-uncased",
        help="The name of the Hugging Face pre-trained transformer model to use for encoding."
    )
    parser.add_argument(
        "--num_motifs",
        type=int,
        default=50,
        help="The number of design motifs to discover (k for k-means).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for generating embeddings.",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random state for k-means for reproducibility."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="The device to use for computation (e.g., 'cuda', 'cpu')."
    )
    return parser


def load_design_traces(data_dir: str) -> List[str]:
    """Loads design traces from a directory of text files.

    Scans the given directory for files with a '.txt' extension and reads
    the content of each file as a single design trace.

    Args:
        data_dir (str): The path to the directory containing design trace files.

    Returns:
        List[str]: A list of design traces. Returns an empty list if the
                   directory does not exist or contains no .txt files.
    """
    data_path = Path(data_dir)
    if not data_path.is_dir():
        logging.error(f"Data directory not found: {data_dir}")
        return []

    logging.info(f"Loading design traces from: {data_dir}")
    design_traces: List[str] = []
    for file_path in data_path.glob("*.txt"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                design_traces.append(f.read().strip())
        except Exception as e:
            logging.warning(f"Could not read file {file_path}: {e}")

    logging.info(f"Successfully loaded {len(design_traces)} design traces.")
    return design_traces


def discover_and_save_motifs(
    model: MotifEncoder,
    design_traces: List[str],
    num_motifs: int,
    batch_size: int,
    output_dir: str,
    random_state: int,
) -> None:
    """Generates embeddings and performs clustering to find and save motifs.

    Uses the encoder to generate an embedding for each design trace in batches.
    Then, it runs k-means clustering on these embeddings. Finally, it saves the
    encoder (with the fitted k-means model) and the cluster assignments to the
    specified output directory.

    Args:
        model (MotifEncoder): The initialized MotifEncoder.
        design_traces (List[str]): The dataset of design traces.
        num_motifs (int): The target number of motifs to discover (k).
        batch_size (int): The batch size for processing embeddings.
        output_dir (str): The directory where artifacts will be saved.
        random_state (int): The random state for K-Means reproducibility.
    """
    model.eval()
    all_embeddings = []

    logging.info(f"Generating embeddings for {len(design_traces)} design traces...")
    with torch.no_grad():
        for i in tqdm(range(0, len(design_traces), batch_size), desc="Encoding Traces"):
            batch_traces = design_traces[i:i + batch_size]
            batch_embeddings = model.encode(batch_traces)
            all_embeddings.append(batch_embeddings.cpu())

    embeddings_tensor = torch.cat(all_embeddings, dim=0)
    logging.info(f"Embeddings generated with shape: {embeddings_tensor.shape}")

    logging.info(f"Discovering {num_motifs} motifs using k-means clustering...")
    model.discover_motifs(embeddings_tensor, n_clusters=num_motifs, random_state=random_state)
    logging.info("K-means clustering complete. Motif centroids have been identified.")

    # Get assignments for analysis
    assignments = model.get_motif_assignments(embeddings_tensor)

    # Save the artifacts
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logging.info(f"Saving artifacts to: {output_dir}")
    model.save(output_path)
    logging.info(f"Saved MotifEncoder model and k-means state to '{output_dir}'.")

    # Save cluster assignments for analysis
    assignments_path = output_path / "motif_assignments.pt"
    torch.save(assignments.cpu(), assignments_path)
    logging.info(f"Saved design trace cluster assignments to: {assignments_path}")


def main() -> None:
    """Main function to run the motif discovery pipeline."""
    parser = setup_arg_parser()
    args = parser.parse_args()

    # Load and process data
    design_traces = load_design_traces(args.data_dir)
    if not design_traces:
        logging.error("No design traces loaded. Exiting.")
        return

    # Initialize the model
    logging.info(f"Initializing MotifEncoder with model '{args.model_name}' on device '{args.device}'.")
    try:
        model = MotifEncoder(model_name_or_path=args.model_name, device=args.device)
    except Exception as e:
        logging.error(f"Failed to initialize MotifEncoder: {e}")
        return

    # In this pipeline, the encoder is pre-trained and not fine-tuned.
    # The "training" part refers to fitting the K-Means model.
    logging.info("Using pre-trained weights for the encoder. No fine-tuning will be performed.")

    # Discover motifs and save all artifacts
    discover_and_save_motifs(
        model=model,
        design_traces=design_traces,
        num_motifs=args.num_motifs,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        random_state=args.random_state
    )

    logging.info("\nMotif discovery process completed successfully.")


if __name__ == "__main__":
    main()