import logging
import os
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics.pairwise import euclidean_distances
from tqdm import tqdm

# Internal Project Imports
# It's assumed that the project is installed in editable mode or the path is configured.
# To run this script directly, you might need to adjust sys.path.
# For example:
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.lspo_3d import config
from lspo_3d.data.dataset import CSGDataset
from lspo_3d.data.tokenizer import CSGTokenizer
from lspo_3d.models.encoder import CSGEncoder
from lspo_3d.models.generator import CSGGenerator
from lspo_3d.utils import setup_logging

# Setup basic logging
setup_logging()
LOGGER = logging.getLogger(__name__)


class MotifSupervisedDataset(Dataset):
    """
    A PyTorch Dataset for supervised training of the CSGGenerator.

    This dataset holds pairs of (motif_id, tokenized_scad_sequence), which are
    used to train the generator to reconstruct a design given its abstract
    motif ID.
    """

    def __init__(self, data: List[Tuple[int, torch.Tensor]]):
        """
        Initializes the MotifSupervisedDataset.

        Args:
            data (List[Tuple[int, torch.Tensor]]): A list where each element is a
                tuple containing a motif ID (int) and the corresponding
                tokenized OpenSCAD script (torch.Tensor).
        """
        self.data = data
        LOGGER.info(f"Created MotifSupervisedDataset with {len(self.data)} samples.")

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: The size of the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a sample from the dataset at the given index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the motif ID
                as a tensor and the tokenized sequence.
        """
        motif_id, sequence = self.data[idx]
        return torch.tensor(motif_id, dtype=torch.long), sequence


def map_designs_to_motifs(
    encoder: CSGEncoder,
    centroids: np.ndarray,
    corpus_dataset: CSGDataset,
    device: torch.device
) -> List[Tuple[int, torch.Tensor]]:
    """
    Maps each design in the corpus to its corresponding motif ID.

    This function iterates through the entire design corpus, generates an
    embedding for each design using the pre-trained encoder, and finds the
    closest motif centroid to determine the design's motif ID.

    Args:
        encoder (CSGEncoder): The pre-trained CSG Encoder model.
        centroids (np.ndarray): The cluster centroids representing the motifs.
        corpus_dataset (CSGDataset): The dataset containing all raw designs.
        device (torch.device): The device (CPU or GPU) to perform computations on.

    Returns:
        List[Tuple[int, torch.Tensor]]: A list of tuples, where each tuple
            contains a motif ID and its corresponding tokenized sequence.
    """
    encoder.eval()
    mapped_data = []

    # Use a DataLoader for efficient batching during inference
    data_loader = DataLoader(
        dataset=corpus_dataset,
        batch_size=config.GENERATOR_PRETRAIN_BATCH_SIZE, # Reuse batch size for convenience
        shuffle=False
    )

    with torch.no_grad():
        for sequences_batch in tqdm(data_loader, desc="Mapping designs to motifs"):
            sequences_batch = sequences_batch.to(device)

            # Get embeddings from the encoder
            embeddings_batch = encoder(sequences_batch)
            embeddings_np = embeddings_batch.cpu().numpy()

            # Calculate distances to all centroids
            # distances shape: (batch_size, num_centroids)
            distances = euclidean_distances(embeddings_np, centroids)

            # Find the index of the closest centroid for each embedding
            # motif_ids shape: (batch_size,)
            motif_ids = np.argmin(distances, axis=1)

            # Store pairs of (motif_id, original_sequence)
            for i in range(len(motif_ids)):
                motif_id = motif_ids[i]
                # Move original sequence back to CPU to store in a list
                original_sequence = sequences_batch[i].cpu()
                mapped_data.append((motif_id, original_sequence))

    return mapped_data


def train_generator_supervised(
    generator: CSGGenerator,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    epochs: int,
    device: torch.device
) -> None:
    """
    Executes the supervised training loop for the CSG Generator.

    Args:
        generator (CSGGenerator): The generator model to be trained.
        data_loader (DataLoader): DataLoader providing batches of
            (motif_id, target_sequence) pairs.
        optimizer (torch.optim.Optimizer): The optimizer for updating model weights.
        criterion (nn.Module): The loss function (e.g., CrossEntropyLoss).
        epochs (int): The total number of training epochs.
        device (torch.device): The device (CPU or GPU) to train on.
    """
    generator.train()
    vocab_size = generator.output_vocab_size

    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(data_loader, desc=f"Epoch {epoch + 1}/{epochs}")

        for batch_idx, (motif_ids, target_sequences) in enumerate(progress_bar):
            motif_ids = motif_ids.to(device)
            target_sequences = target_sequences.to(device)

            optimizer.zero_grad()

            # The generator receives the motif and the full target sequence.
            # Internally, it uses causal masking to predict the next token at each position.
            output_logits = generator(motif_ids, target_sequences)

            # To calculate CrossEntropyLoss, we need to reshape the logits and targets.
            # Logits: from (batch, seq_len, vocab_size) to (batch * seq_len, vocab_size)
            # Targets: from (batch, seq_len) to (batch * seq_len)
            loss = criterion(output_logits.view(-1, vocab_size), target_sequences.view(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)  # Gradient clipping
            optimizer.step()

            total_loss += loss.item()

            # Update progress bar with current loss
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / len(data_loader)
        perplexity = np.exp(avg_loss)
        LOGGER.info(f"Epoch {epoch + 1}/{epochs} | Average Loss: {avg_loss:.4f} | Perplexity: {perplexity:.4f}")

def main():
    """
    Main function to orchestrate the pre-training of the CSG Generator.

    The script performs the following steps:
    1. Loads configuration and sets up the execution device.
    2. Loads the pre-trained tokenizer, encoder model, and motif centroids
       which were created by the '01_create_latent_space.py' script.
    3. Iterates through the design corpus to create a mapping from each
       design to its corresponding abstract motif ID.
    4. Creates a custom PyTorch Dataset and DataLoader for this supervised task.
    5. Initializes and trains the CSG Generator model.
    6. Saves the pre-trained generator model weights for later use.
    """
    LOGGER.info("Starting supervised pre-training of the CSG Generator.")

    # 1. Load configuration and setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info(f"Using device: {device}")

    # 2. Load prerequisites from the latent space creation step
    LOGGER.info("Loading tokenizer, encoder, and motif centroids...")
    try:
        if not os.path.exists(config.TOKENIZER_PATH) or \
           not os.path.exists(config.ENCODER_MODEL_PATH) or \
           not os.path.exists(config.MOTIF_CENTROIDS_PATH):
            raise FileNotFoundError("One or more required files (tokenizer, encoder, centroids) not found.")

        tokenizer = CSGTokenizer.load(filepath=config.TOKENIZER_PATH)
        encoder = CSGEncoder(
            vocab_size=tokenizer.get_vocab_size(),
            d_model=config.ENCODER_D_MODEL,
            n_head=config.ENCODER_N_HEAD,
            n_layers=config.ENCODER_N_LAYERS,
            dropout=config.ENCODER_DROPOUT,
            max_seq_len=config.MAX_SEQ_LEN
        )
        encoder.load_state_dict(torch.load(config.ENCODER_MODEL_PATH, map_location=device))
        encoder.to(device)
        encoder.eval()

        motif_centroids = np.load(config.MOTIF_CENTROIDS_PATH)
        LOGGER.info(f"Loaded {len(motif_centroids)} motif centroids.")
    except FileNotFoundError as e:
        LOGGER.error(f"Failed to load prerequisites: {e}. "
                     "Please run 'scripts/01_create_latent_space.py' first.")
        return
    except Exception as e:
        LOGGER.error(f"An unexpected error occurred while loading prerequisites: {e}")
        return

    # 3. Create the mapping from designs to motifs
    LOGGER.info("Mapping all designs in the corpus to their respective motifs...")
    corpus_dataset = CSGDataset(
        corpus_dir=config.CORPUS_DIR,
        tokenizer=tokenizer,
        max_seq_len=config.MAX_SEQ_LEN
    )
    supervised_data = map_designs_to_motifs(encoder, motif_centroids, corpus_dataset, device)

    # 4. Create Dataset and DataLoader
    LOGGER.info("Creating DataLoader for supervised training.")
    training_dataset = MotifSupervisedDataset(data=supervised_data)
    train_loader = DataLoader(
        dataset=training_dataset,
        batch_size=config.GENERATOR_PRETRAIN_BATCH_SIZE,
        shuffle=True
    )

    # 5. Initialize the Generator, optimizer, and loss function
    LOGGER.info("Initializing CSG Generator model.")
    generator = CSGGenerator(
        motif_vocab_size=len(motif_centroids),
        output_vocab_size=tokenizer.get_vocab_size(),
        d_model=config.GENERATOR_D_MODEL,
        n_head=config.GENERATOR_N_HEAD,
        n_layers=config.GENERATOR_N_LAYERS,
        max_seq_len=config.MAX_SEQ_LEN,
        dropout=config.GENERATOR_DROPOUT
    )
    generator.to(device)

    optimizer = AdamW(generator.parameters(), lr=config.GENERATOR_PRETRAIN_LR)
    # Ignore padding index in loss calculation to not penalize the model for padding.
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    # 6. Run the training loop
    LOGGER.info("Starting training...")
    train_generator_supervised(
        generator=generator,
        data_loader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        epochs=config.GENERATOR_PRETRAIN_EPOCHS,
        device=device
    )

    # 7. Save the trained model
    LOGGER.info("Training complete. Saving pre-trained generator model.")
    try:
        os.makedirs(os.path.dirname(config.GENERATOR_PRETRAINED_PATH), exist_ok=True)
        torch.save(generator.state_dict(), config.GENERATOR_PRETRAINED_PATH)
        LOGGER.info(f"Model saved to {config.GENERATOR_PRETRAINED_PATH}")
    except IOError as e:
        LOGGER.error(f"Could not save the model to '{config.GENERATOR_PRETRAINED_PATH}': {e}")


if __name__ == "__main__":
    main()