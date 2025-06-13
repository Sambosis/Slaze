# scripts/01_create_latent_space.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))


"""
Standalone script to create the latent strategy space for 3D object generation.

This script performs the following sequential steps:
1.  Loads the configuration and sets up the environment (device, logging).
2.  Initializes the CSGTokenizer and the CSGDataset to load the corpus of
    .scad files.
3.  Initializes the CSGEncoder model, optimizer, and loss function.
4.  Trains the CSGEncoder on the tokenized .scad scripts to learn meaningful
    vector representations of 3D designs.
5.  Uses the trained encoder to generate embeddings for every design in the
    corpus.
6.  Applies k-means clustering to the collection of embeddings to identify a
    predefined number of "design motifs". The centroids of these clusters
    become the canonical representations of the motifs.
7.  Saves the trained CSGEncoder model state dictionary and the calculated
    cluster centroids (motifs) to disk for later use by the generator and
    RL agent.
"""

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
from tqdm import tqdm

# Internal project imports
from lspo_3d import config, utils
from lspo_3d.data.dataset import CSGDataset
from lspo_3d.data.tokenizer import CSGTokenizer
from lspo_3d.models.encoder import CSGEncoder


def train_encoder(
    encoder: CSGEncoder,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    num_epochs: int,
    vocab_size: int
) -> CSGEncoder:
    """Trains the CSGEncoder model.

    This function assumes a language-model-like training objective where the model
    predicts the next token in a sequence. The `CSGEncoder` is expected to
    return logits for this task.

    Args:
        encoder (CSGEncoder): The encoder model to be trained.
        dataloader (DataLoader): DataLoader providing batches of tokenized sequences.
        optimizer (torch.optim.Optimizer): The optimization algorithm.
        criterion (torch.nn.Module): The loss function (e.g., CrossEntropyLoss).
        device (torch.device): The device (CPU or CUDA) to train on.
        num_epochs (int): The number of complete passes through the dataset.
        vocab_size (int): The size of the tokenizer's vocabulary.

    Returns:
        CSGEncoder: The trained encoder model.
    """
    logging.info(f"Starting encoder training for {num_epochs} epochs on {device}.")
    encoder.to(device)
    encoder.train()

    for epoch in range(num_epochs):
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for batch in pbar:
            # The dataset returns a single tensor of token IDs per item
            tokens = batch.to(device)

            # Prepare inputs and targets for next-token prediction
            # Inputs: sequence[:-1], Targets: sequence[1:]
            inputs = tokens[:, :-1]
            targets = tokens[:, 1:].reshape(-1) # Flatten for CrossEntropyLoss

            optimizer.zero_grad()

            # The encoder's forward pass for training should return logits.
            # We assume it outputs of shape [batch_size, seq_len, vocab_size].
            outputs, _ = encoder(inputs) # Assume returns (logits, final_embedding)
            
            # Reshape outputs to [batch_size * seq_len, vocab_size] for criterion
            outputs_flat = outputs.view(-1, vocab_size)

            loss = criterion(outputs_flat, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(dataloader)
        logging.info(f"Epoch {epoch + 1} complete. Average Loss: {avg_loss:.4f}")

    logging.info("Encoder training completed.")
    return encoder


def generate_embeddings(
    encoder: CSGEncoder, dataloader: DataLoader, device: torch.device
) -> np.ndarray:
    """Generates embeddings for the entire dataset using the trained encoder.

    Args:
        encoder (CSGEncoder): The trained encoder model.
        dataloader (DataLoader): DataLoader for the dataset.
        device (torch.device): The device (CPU or CUDA) to run inference on.

    Returns:
        np.ndarray: A 2D numpy array where each row is the embedding for a design.
                    Shape: (num_designs, embedding_dim).
    """
    logging.info("Generating embeddings for the entire dataset.")
    encoder.to(device)
    encoder.eval()

    all_embeddings = []
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Generating Embeddings")
        for batch in pbar:
            tokens = batch.to(device)
            # The encoder's forward pass should also be able to return a single
            # summary embedding vector for the entire sequence.
            # We assume the second element of the returned tuple is this embedding.
            _, embedding = encoder(tokens) # Shape: [batch_size, d_model]
            all_embeddings.append(embedding.cpu())

    if not all_embeddings:
        logging.warning("No embeddings were generated. The dataset might be empty.")
        return np.array([])
        
    embeddings_tensor = torch.cat(all_embeddings, dim=0)
    logging.info(f"Embeddings generation complete. Generated {embeddings_tensor.shape[0]} embeddings.")
    return embeddings_tensor.numpy()


def perform_clustering(
    embeddings: np.ndarray, num_clusters: int
) -> Tuple[KMeans, np.ndarray]:
    """Performs k-means clustering on the generated embeddings.

    Args:
        embeddings (np.ndarray): A 2D array of design embeddings.
        num_clusters (int): The number of clusters (motifs) to create.

    Returns:
        Tuple[KMeans, np.ndarray]: A tuple containing:
            - The fitted KMeans model instance.
            - A numpy array of the cluster centroids (the motifs).
              Shape: (num_clusters, embedding_dim).
    """
    if embeddings.shape[0] < num_clusters:
        raise ValueError(
            f"Number of samples ({embeddings.shape[0]}) must be >= number of clusters ({num_clusters})."
        )
        
    logging.info(f"Performing K-means clustering to find {num_clusters} motifs.")
    kmeans = KMeans(
        n_clusters=num_clusters,
        random_state=config.SEED,
        n_init="auto",
        verbose=0
    )
    
    kmeans.fit(embeddings)

    centroids = kmeans.cluster_centers_
    logging.info(f"Clustering complete. Found {len(centroids)} motifs from {embeddings.shape[0]} samples.")
    return kmeans, centroids


def save_artifacts(
    encoder: CSGEncoder, centroids: np.ndarray, output_dir: Path
) -> None:
    """Saves the trained encoder and the cluster centroids to disk.

    Args:
        encoder (CSGEncoder): The trained CSGEncoder model.
        centroids (np.ndarray): The array of cluster centroids (motifs).
        output_dir (Path): The directory to save the artifacts in.
    """
    logging.info(f"Saving artifacts to {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    encoder_path = output_dir / config.ENCODER_FILENAME
    centroids_path = output_dir / config.MOTIFS_FILENAME

    torch.save(encoder.state_dict(), encoder_path)
    np.save(centroids_path, centroids)

    logging.info(f"Encoder saved to {encoder_path}")
    logging.info(f"Motifs saved to {centroids_path}")


def main() -> None:
    """Main function to run the latent space creation pipeline."""
    # 1. Setup
    utils.setup_logging()
    logging.info("--- Starting Script: 01_create_latent_space ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Load Data
    logging.info(f"Loading tokenizer and dataset from {config.SCAD_CORPUS_DIR}...")
    if config.TOKENIZER_PATH.exists():
        logging.info("Found existing tokenizer. Loading it.")
        tokenizer = CSGTokenizer.load(str(config.TOKENIZER_PATH))
    else:
        logging.info("No tokenizer found. Training a new one.")
        tokenizer = CSGTokenizer()
        files = [str(p) for p in config.SCAD_CORPUS_DIR.glob("**/*.scad")]
        if not files:
            logging.error(f"No .scad files found in {config.SCAD_CORPUS_DIR} for training the tokenizer.")
            return
        tokenizer.train(files)
        tokenizer.save(str(config.TOKENIZER_PATH))

    dataset = CSGDataset(
        data_dir=str(config.SCAD_CORPUS_DIR),
        tokenizer=tokenizer,
        max_seq_length=config.MAX_SEQUENCE_LENGTH,
    )
    if len(dataset) == 0:
        logging.error(f"No .scad files found in {config.SCAD_CORPUS_DIR}. Exiting.")
        return
        
    dataloader = DataLoader(
        dataset, batch_size=config.BATCH_SIZE, shuffle=True
    )
    vocab_size = tokenizer.get_vocab_size()

    # 3. Initialize Model and Training Components
    logging.info("Initializing CSGEncoder model...")
    encoder = CSGEncoder(
        vocab_size=vocab_size,
        d_model=config.ENCODER_EMBEDDING_DIM,
        nhead=config.ENCODER_NUM_HEADS,
        nlayers=config.ENCODER_NUM_LAYERS,
        d_hid=config.ENCODER_FF_DIM,
        dropout=config.ENCODER_DROPOUT
    )
    optimizer = torch.optim.Adam(
        encoder.parameters(), lr=config.LEARNING_RATE
    )
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.tokenizer.token_to_id("[PAD]"))

    # 4. Train Encoder
    trained_encoder = train_encoder(
        encoder=encoder,
        dataloader=dataloader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        num_epochs=config.NUM_ENCODER_TRAIN_EPOCHS,
        vocab_size=vocab_size,
    )

    # 5. Generate Embeddings
    inference_dataloader = DataLoader(
        dataset, batch_size=config.BATCH_SIZE, shuffle=False
    )
    embeddings = generate_embeddings(
        encoder=trained_encoder, dataloader=inference_dataloader, device=device
    )

    if embeddings.size == 0:
        logging.error("Failed to generate embeddings. Aborting.")
        return

    # 6. Perform Clustering
    try:
        _, motifs = perform_clustering(
            embeddings=embeddings, num_clusters=config.NUM_MOTIFS
        )
    except ValueError as e:
        logging.error(f"Clustering failed: {e}")
        return

    # 7. Save Artifacts
    save_artifacts(encoder=trained_encoder, centroids=motifs, output_dir=config.ARTIFACTS_DIR)

    logging.info("--- Script 01_create_latent_space Finished Successfully ---")


if __name__ == "__main__":
    main()