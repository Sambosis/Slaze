"""
Defines the Transformer-based encoder for creating embeddings from CSG
sequences and the k-means clustering logic to discover design motifs.
"""

# External Imports
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from sklearn.cluster import KMeans
import joblib
from pathlib import Path
import logging

# Typing Imports
from typing import List, Optional, Union

logger = logging.getLogger(__name__)


class MotifEncoder(nn.Module):
    """
    Encapsulates a Transformer model for encoding CSG design traces and a
    K-Means model for discovering design motifs from the resulting embeddings.

    This class handles the end-to-end process of converting raw design trace
    strings into a set of discrete, learned "motifs" that represent high-level
    design concepts.

    Attributes:
        model (PreTrainedModel): The underlying Transformer model used for encoding.
        tokenizer (PreTrainedTokenizer): The tokenizer corresponding to the model.
        device (torch.device): The device (e.g., 'cuda' or 'cpu') the model is on.
        kmeans (Optional[KMeans]): The scikit-learn K-Means model. Will be
            initialized after training.
        motif_centroids (Optional[torch.Tensor]): A tensor containing the cluster
            centers, which represent the learned design motifs.
    """

    def __init__(self, model_name_or_path: str, device: Union[str, torch.device] = 'cpu'):
        """
        Initializes the MotifEncoder by loading a pre-trained Transformer model
        and its tokenizer.

        Args:
            model_name_or_path (str): The identifier for a pre-trained model from
                the Hugging Face Hub (e.g., 'bert-base-uncased') or a path to a
                local directory containing model files.
            device (Union[str, torch.device]): The device ('cpu' or 'cuda') to
                run the model on. Defaults to 'cpu'.
        """
        super().__init__()
        self.device = torch.device(device)
        try:
            self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            self.model: PreTrainedModel = AutoModel.from_pretrained(model_name_or_path).to(self.device).eval()
        except OSError as e:
            logger.error(f"Could not load model from '{model_name_or_path}'. Ensure the path is correct or the model exists on Hugging Face Hub.")
            raise e
            
        self.kmeans: Optional[KMeans] = None
        self.motif_centroids: Optional[torch.Tensor] = None
        logger.info(f"MotifEncoder initialized with model '{model_name_or_path}' on device '{self.device}'.")

    def encode(self, design_traces: List[str], pool_strategy: str = 'mean') -> torch.Tensor:
        """
        Encodes a list of CSG design traces into a tensor of embeddings.

        The method tokenizes the input strings, passes them through the transformer
        model, and applies a pooling strategy to obtain a fixed-size vector
        for each trace.

        Args:
            design_traces (List[str]): A list of strings, where each string is a
                CSG operations sequence (e.g., a line of cadquery code).
            pool_strategy (str): The pooling strategy to use on the model's output
                hidden states. Supports 'mean' or 'cls'. Defaults to 'mean'.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, embedding_dim) containing
                the resulting embeddings, located on the instance's device.
        """
        if not design_traces:
            return torch.empty(0, self.model.config.hidden_size, device=self.device)

        # Tokenize the input texts
        encoded_input = self.tokenizer(
            design_traces,
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).to(self.device)

        # Compute token embeddings with no gradient calculation for efficiency
        with torch.no_grad():
            outputs = self.model(**encoded_input)
            token_embeddings = outputs.last_hidden_state

        # Apply pooling strategy
        if pool_strategy == 'cls':
            # Use the embedding of the [CLS] token
            sentence_embeddings = token_embeddings[:, 0]
        elif pool_strategy == 'mean':
            # Perform mean pooling, taking the attention mask into account
            # to correctly average over non-padding tokens.
            attention_mask = encoded_input['attention_mask']
            mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            sentence_embeddings = sum_embeddings / sum_mask
        else:
            raise ValueError(f"Unsupported pool_strategy: '{pool_strategy}'. Use 'mean' or 'cls'.")

        return sentence_embeddings

    def discover_motifs(self, embeddings: torch.Tensor, n_clusters: int, random_state: Optional[int] = None):
        """
        Performs K-Means clustering on the provided embeddings to find motifs.

        This method fits a K-Means model to the data and stores the resulting
        model and its cluster centers (centroids). The centroids are considered
        the learned design motifs.

        Args:
            embeddings (torch.Tensor): A tensor of embeddings of shape
                (n_samples, embedding_dim) to be clustered.
            n_clusters (int): The number of motifs (k) to find.
            random_state (Optional[int]): The random state for K-Means reproducibility.
                Defaults to None.
        """
        logger.info(f"Discovering {n_clusters} motifs using K-Means clustering...")
        # Move embeddings to CPU and convert to numpy for scikit-learn
        embeddings_np = embeddings.cpu().numpy()

        # Initialize and fit K-Means model
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')
        kmeans.fit(embeddings_np)
        
        self.kmeans = kmeans
        
        # Store cluster centers as torch tensor on the correct device
        centroids_np = self.kmeans.cluster_centers_
        self.motif_centroids = torch.from_numpy(centroids_np).to(self.device)
        logger.info("Motif discovery complete. Centroids stored.")

    def get_motif_assignments(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Assigns each embedding in a batch to the nearest learned motif.

        Args:
            embeddings (torch.Tensor): A tensor of embeddings of shape
                (batch_size, embedding_dim).

        Returns:
            torch.Tensor: A tensor of shape (batch_size,) containing the integer
                ID (cluster index) for each embedding's corresponding motif.

        Raises:
            RuntimeError: If the K-Means model has not been trained yet.
        """
        if self.kmeans is None:
            raise RuntimeError("Cannot assign motifs. The K-Means model has not been trained. Call `discover_motifs` first.")
            
        # Convert tensor to CPU numpy array for prediction
        embeddings_np = embeddings.cpu().numpy()
        
        # Predict cluster labels
        labels = self.kmeans.predict(embeddings_np)
        
        # Convert labels back to a torch tensor on the target device
        return torch.from_numpy(labels).to(self.device)

    def save(self, save_directory: Union[str, Path]):
        """
        Saves the entire MotifEncoder state to a directory.

        This includes the fine-tuned Transformer model, its tokenizer, and the
        fitted K-Means model.

        Args:
            save_directory (Union[str, Path]): The path to the directory where
                the state will be saved.
        """
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        # Save transformer model and tokenizer
        self.model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)
        logger.info(f"Transformer model and tokenizer saved to '{save_directory}'.")

        # Save KMeans model if it exists
        if self.kmeans:
            kmeans_path = save_directory / "kmeans.joblib"
            joblib.dump(self.kmeans, kmeans_path)
            logger.info(f"K-Means model saved to '{kmeans_path}'.")

    @classmethod
    def load(cls, load_directory: Union[str, Path], device: Union[str, torch.device] = 'cpu') -> "MotifEncoder":
        """
        Loads a complete MotifEncoder instance from a directory.

        Args:
            load_directory (Union[str, Path]): The directory from which to load the state.
            device (Union[str, torch.device]): The device to load the model onto.
                Defaults to 'cpu'.

        Returns:
            MotifEncoder: An instance of MotifEncoder with the loaded state.
        """
        load_directory = Path(load_directory)
        if not load_directory.isdir():
            raise FileNotFoundError(f"Load directory not found: {load_directory}")

        # Instantiate the class, which loads the transformer model and tokenizer
        instance = cls(model_name_or_path=str(load_directory), device=device)
        
        # Check for and load the KMeans model
        kmeans_path = load_directory / "kmeans.joblib"
        if kmeans_path.exists():
            instance.kmeans = joblib.load(kmeans_path)
            logger.info(f"Loaded K-Means model from '{kmeans_path}'.")
            
            # If kmeans was loaded, re-establish the centroids tensor
            centroids_np = instance.kmeans.cluster_centers_
            instance.motif_centroids = torch.from_numpy(centroids_np).to(instance.device)
            logger.info("Motif centroids updated from loaded K-Means model.")
        else:
            logger.warning(f"No K-Means model ('kmeans.joblib') found in '{load_directory}'.")

        return instance