# C:\Users\Machine81\Slazy\repo\stls\3d_lspo\models\motif_encoder.py

"""
Defines the MotifEncoder class responsible for converting CSG sequences into
embeddings and discovering abstract design motifs through clustering.

This module contains the core components for Phase 1 of the 3D-LSPO project:
1.  A Transformer-based model to generate semantic embeddings from sequences
    of CAD operations (design traces).
2.  A k-means clustering algorithm to group these embeddings into a discrete
    set of "design motifs", where each cluster centroid represents a motif.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Union

import joblib
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.exceptions import NotFittedError
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer


class MotifEncoder:
    """
    Encapsulates the Transformer encoder and k-means clustering model.

    This class provides a complete pipeline for transforming raw CSG operation
    sequences into discrete motif IDs. It handles tokenization, embedding
    generation, motif discovery via clustering, and prediction for new sequences.
    It also includes methods for saving and loading the entire trained state.

    Attributes:
        model_name (str): The name of the pretrained transformer model from
                          Hugging Face hub.
        num_motifs (int): The number of clusters (motifs) to discover.
        device (torch.device): The device (CPU or CUDA) to run the model on.
        tokenizer (PreTrainedTokenizer): The tokenizer for the transformer model.
        encoder_model (PreTrainedModel): The transformer model for encoding.
        kmeans_model (KMeans): The scikit-learn k-means model for clustering.
        motif_centroids (torch.Tensor | None): The centroids of the discovered
                                               clusters, representing the motifs.
    """

    def __init__(self, model_name: str, num_motifs: int):
        """
        Initializes the MotifEncoder with a specified transformer and k-means config.

        Args:
            model_name (str): The identifier for a pretrained model from the
                              Hugging Face Hub (e.g., 'bert-base-uncased').
            num_motifs (int): The number of design motifs to discover, which
                              corresponds to the 'k' in k-means.
        
        Raises:
            ConnectionError: If the specified model cannot be downloaded from the
                             Hugging Face Hub.
        """
        self.model_name: str = model_name
        self.num_motifs: int = num_motifs
        self.device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"Using device: {self.device}")

        try:
            self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name)
            self.encoder_model: PreTrainedModel = AutoModel.from_pretrained(model_name).to(
                self.device
            )
        except OSError as e:
            raise ConnectionError(
                f"Could not download model '{model_name}' from Hugging Face Hub. "
                "Please check your internet connection and the model identifier."
            ) from e

        # Set n_init explicitly to 10 to suppress FutureWarning and use the new default.
        self.kmeans_model: KMeans = KMeans(
            n_clusters=self.num_motifs, random_state=42, n_init=10
        )
        self.motif_centroids: torch.Tensor | None = None

    def _mean_pooling(self, model_output, attention_mask: torch.Tensor) -> torch.Tensor:
        """Helper function for mean pooling of token embeddings."""
        token_embeddings = model_output[0]  # First element of model_output is last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def encode(self, csg_sequences: List[str]) -> torch.Tensor:
        """
        Encodes a batch of CSG sequences into fixed-size embeddings.

        This method tokenizes the input strings, passes them through the
        transformer model, and performs mean pooling on the last hidden state
        to get a sentence-level embedding for each sequence.

        Args:
            csg_sequences (List[str]): A list of strings, where each string is a
                                       design trace (sequence of CSG operations).

        Returns:
            torch.Tensor: A 2D tensor of shape (batch_size, embedding_dim)
                          containing the embeddings for the input sequences,
                          detached from the computation graph and on the CPU.
        """
        if not csg_sequences:
            return torch.empty(0, self.encoder_model.config.hidden_size)

        inputs = self.tokenizer(
            csg_sequences,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=512
        )
        inputs = {key: val.to(self.device) for key, val in inputs.items()}

        self.encoder_model.eval()
        with torch.no_grad():
            outputs = self.encoder_model(**inputs)

        embeddings = self._mean_pooling(outputs, inputs['attention_mask'])
        
        return embeddings.detach().cpu()

    def discover_motifs(self, embeddings: np.ndarray) -> None:
        """
        Fits the k-means model to the embeddings to discover motifs.

        This method performs k-means clustering on the provided embeddings.
        After fitting, it stores the cluster centers as the canonical
        motif representations.

        Args:
            embeddings (np.ndarray): A 2D numpy array of embeddings generated by the
                                     `encode` method.
        
        Raises:
            ValueError: If the number of embeddings is less than the number of clusters.
        """
        if embeddings.shape[0] < self.num_motifs:
            raise ValueError(
                f"Number of samples ({embeddings.shape[0]}) must be >= number of clusters ({self.num_motifs})."
            )
        
        print(f"Fitting KMeans model with {self.num_motifs} clusters...")
        self.kmeans_model.fit(embeddings)
        
        self.motif_centroids = torch.from_numpy(self.kmeans_model.cluster_centers_).to(self.device)
        print("KMeans fitting complete. Motif centroids discovered.")

    def get_motif_ids(self, csg_sequences: List[str]) -> np.ndarray:
        """
        Predicts the motif ID for a batch of new CSG sequences.

        This method first encodes the sequences to get their embeddings and then
        uses the trained k-means model to predict which cluster (motif) each
        sequence belongs to.

        Args:
            csg_sequences (List[str]): A list of CSG sequences to classify.

        Returns:
            np.ndarray: A 1D numpy array of integer motif IDs corresponding to
                        the input sequences.
                        
        Raises:
            NotFittedError: If the k-means model has not been fitted yet (i.e.,
                            `discover_motifs` has not been called).
        """
        try:
            # Check if kmeans is fitted by accessing an attribute that exists after fitting
            _ = self.kmeans_model.cluster_centers_
        except (AttributeError, NotFittedError) as e:
            raise NotFittedError(
                "The KMeans model has not been fitted yet. Call `discover_motifs` with embeddings first."
            ) from e

        embeddings_tensor = self.encode(csg_sequences)
        embeddings_numpy = embeddings_tensor.numpy()

        return self.kmeans_model.predict(embeddings_numpy)

    def save(self, save_directory: Union[str, Path]) -> None:
        """
        Saves the trained encoder, tokenizer, and k-means model to a directory.

        The transformer model and tokenizer are saved using the Hugging Face
        `save_pretrained` method. The k-means model is saved using joblib. A config
        file is also saved to preserve hyperparameters.

        Args:
            save_directory (Union[str, Path]): The path to the directory where
                                              the models will be saved.
        """
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Saving encoder model and tokenizer to {save_dir}...")
        self.encoder_model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        
        kmeans_path = save_dir / "kmeans_model.joblib"
        print(f"Saving KMeans model to {kmeans_path}...")
        joblib.dump(self.kmeans_model, kmeans_path)

        config_path = save_dir / "config.json"
        config = {
            "model_name": self.model_name,
            "num_motifs": self.num_motifs
        }
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4)
            
        print(f"MotifEncoder saved successfully to {save_dir}.")

    @classmethod
    def load(cls, load_directory: Union[str, Path]) -> MotifEncoder:
        """
        Loads a trained MotifEncoder from a directory.

        This class method reconstructs a MotifEncoder instance by loading the
        saved transformer model, tokenizer, k-means model, and config from the
        specified directory.

        Args:
            load_directory (Union[str, Path]): The path to the directory
                                              containing the saved models.

        Returns:
            MotifEncoder: A new instance of the class with loaded weights and states.
        
        Raises:
            FileNotFoundError: If the directory or required files are not found.
        """
        load_dir = Path(load_directory)
        config_path = load_dir / "config.json"
        kmeans_path = load_dir / "kmeans_model.joblib"

        if not load_dir.is_dir() or not config_path.exists() or not kmeans_path.exists():
            raise FileNotFoundError(f"Load directory '{load_dir}' is invalid or missing required files "
                                    "(config.json, kmeans_model.joblib, etc.).")

        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        model_name = config["model_name"]
        num_motifs = config["num_motifs"]
        
        print(f"Loading MotifEncoder with model='{model_name}' and num_motifs={num_motifs}...")
        instance = cls(model_name, num_motifs)
        
        instance.encoder_model = AutoModel.from_pretrained(load_dir).to(instance.device)
        instance.tokenizer = AutoTokenizer.from_pretrained(load_dir)
        instance.kmeans_model = joblib.load(kmeans_path)
        
        try:
            instance.motif_centroids = torch.from_numpy(
                instance.kmeans_model.cluster_centers_
            ).to(instance.device)
        except (AttributeError, NotFittedError):
            instance.motif_centroids = None
            print("Warning: Loaded KMeans model has not been fitted. Motifs are not discovered.")
        
        print(f"MotifEncoder loaded successfully from {load_dir}.")
        return instance