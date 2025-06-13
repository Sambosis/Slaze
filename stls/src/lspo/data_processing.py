# src/lspo/data_processing.py

"""
Contains functions for data loading, processing, and latent space creation.

This module is responsible for the initial data pipeline stages:
1. Loading raw 3D model data, represented as cadquery command sequences.
2. Embedding these textual sequences into a high-dimensional vector space using
   a pre-trained sentence transformer model.
3. Clustering these embeddings using k-means to discover abstract "design motifs".
4. Saving the resulting cluster centroids (the motifs) and labeled data for
   use by the training pipeline.
"""

import logging
from pathlib import Path
from typing import List, Tuple, Optional

import joblib
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

# Internal project imports
import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_raw_cad_sequences(raw_data_dir: Path) -> List[str]:
    """
    Loads raw cadquery command sequences from a directory.

    This function iterates through all '.py' files in the given directory,
    reads their content, and returns a list of strings, where each string
    is the command sequence for a single 3D model.

    Args:
        raw_data_dir (Path): The path to the directory containing the raw
                             cadquery sequence files.

    Returns:
        List[str]: A list of strings, where each string is a script of
                   cadquery commands.
    """
    logging.info(f"Loading raw cadquery sequences from: {raw_data_dir}")
    sequences = []
    file_paths = list(raw_data_dir.glob("*.py"))

    if not file_paths:
        logging.warning(f"No '.py' files found in directory: {raw_data_dir}")
        return []

    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                sequences.append(f.read())
        except IOError as e:
            logging.error(f"Could not read file {file_path}: {e}")

    logging.info(f"Successfully loaded {len(sequences)} cadquery sequences.")
    return sequences


def embed_sequences(sequences: List[str], model_name: str, batch_size: int) -> np.ndarray:
    """
    Encodes a list of cadquery sequences into numerical embeddings.

    Uses a pre-trained SentenceTransformer model to convert the textual
    representation of cadquery scripts into fixed-size dense vectors.

    Args:
        sequences (List[str]): The list of cadquery command sequences to embed.
        model_name (str): The name of the sentence-transformer model to use.
        batch_size (int): The batch size for processing embeddings.

    Returns:
        np.ndarray: A 2D numpy array of shape (num_sequences, embedding_dim)
                    containing the vector embeddings.
    """
    if not sequences:
        logging.warning("Received an empty list of sequences to embed. Returning empty array.")
        return np.array([])
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Initializing SentenceTransformer model '{model_name}' on device '{device}'.")
    model = SentenceTransformer(model_name, device=device)

    logging.info(f"Embedding {len(sequences)} sequences (batch size: {batch_size})...")
    embeddings = model.encode(
        sequences,
        show_progress_bar=True,
        batch_size=batch_size,
        convert_to_numpy=True
    )
    logging.info(f"Embedding complete. Created embeddings of shape: {embeddings.shape}")
    return embeddings


def perform_clustering(embeddings: np.ndarray, num_clusters: int, random_seed: int) -> Tuple[KMeans, np.ndarray]:
    """
    Performs k-means clustering on the sequence embeddings.

    This function groups the vector embeddings into a specified number of
    clusters. The centroids of these clusters represent the abstract "design motifs".

    Args:
        embeddings (np.ndarray): The 2D array of embeddings from the embed_sequences function.
        num_clusters (int): The number of clusters (motifs) to create.
        random_seed (int): A seed for the random number generator for reproducibility.

    Returns:
        Tuple[KMeans, np.ndarray]: A tuple containing:
            - The fitted scikit-learn KMeans model object. This object holds the
              cluster centroids (`cluster_centers_`).
            - A 1D numpy array of cluster labels for each embedding.
    """
    logging.info(f"Performing k-means clustering with {num_clusters} clusters.")
    kmeans = KMeans(
        n_clusters=num_clusters,
        random_state=random_seed,
        n_init=10  # explicit n_init to avoid FutureWarning
    )
    kmeans.fit(embeddings)
    labels = kmeans.labels_
    logging.info("Clustering complete.")
    return kmeans, labels


def save_motif_data(kmeans_model: KMeans, labels: np.ndarray, output_dir: Path) -> None:
    """
    Saves the results of the clustering process to disk.

    This saves the fitted KMeans model (which includes the centroids, i.e., the motifs)
    and the array of labels corresponding to the original sequences. These files
    will be loaded by the trainer.

    Args:
        kmeans_model (KMeans): The fitted KMeans model instance returned by
                               perform_clustering.
        labels (np.ndarray): The array of cluster labels.
        output_dir (Path): The directory where the processed data will be saved.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "kmeans_model.joblib"
    labels_path = output_dir / "sequence_labels.npy"
    centroids_path = output_dir / "motif_centroids.npy"

    logging.info(f"Saving KMeans model to: {model_path}")
    joblib.dump(kmeans_model, model_path)

    logging.info(f"Saving cluster labels to: {labels_path}")
    np.save(labels_path, labels)

    logging.info(f"Saving motif centroids to: {centroids_path}")
    np.save(centroids_path, kmeans_model.cluster_centers_)


def generate_latent_space() -> None:
    """
    Executes the full data processing pipeline to create the latent space.

    This is the main orchestrator function for this module. It performs the
    following steps:
    1. Loads raw cadquery sequences from the data directory.
    2. Embeds these sequences into a vector space.
    3. Clusters the embeddings to find design motifs.
    4. Saves the resulting motif data (centroids and labels) to the
       processed data directory for the trainer to use.
    """
    logging.info("--- Starting Latent Space Generation Pipeline ---")
    cfg = config.get_config()

    # 1. Load raw sequences
    sequences = load_raw_cad_sequences(cfg.paths.raw_models_dir)
    if not sequences:
        logging.error("No sequences loaded. Aborting pipeline.")
        return

    # 2. Embed sequences
    embeddings = embed_sequences(
        sequences,
        model_name=cfg.embedding.model_name,
        batch_size=cfg.embedding.batch_size
    )
    if embeddings.size == 0:
        logging.error("Embedding resulted in an empty array. Aborting pipeline.")
        return

    # 3. Perform clustering
    kmeans_model, labels = perform_clustering(
        embeddings,
        num_clusters=cfg.embedding.num_clusters,
        random_seed=cfg.seed
    )

    # 4. Save results
    save_motif_data(
        kmeans_model=kmeans_model,
        labels=labels,
        output_dir=cfg.paths.processed_sequences_dir
    )

    logging.info("--- Latent Space Generation Pipeline Finished Successfully ---")


if __name__ == "__main__":
    """
    Allows running the data processing pipeline as a standalone script.
    
    Example usage from project root:
        python -m src.lspo.data_processing

    Note: Before running, ensure you have some `.py` files containing CadQuery
    scripts in the `data/raw_models/` directory. You can create dummy files for testing.
    For example, create `data/raw_models/box.py` with content:
    `import cadquery as cq; result = cq.Workplane("XY").box(10, 20, 30)`
    """
    print("Running data processing pipeline as a standalone script...")
    generate_latent_space()
    print("Data processing complete. Check logs and the 'data/processed_sequences' directory.")