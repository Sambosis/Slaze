# -*- coding: utf-8 -*-
"""
Defines the PyTorch Dataset for loading and processing OpenSCAD (.scad) files.

This module contains the CSGDataset class, which is responsible for discovering
.scad files in a directory, reading their contents, and using a tokenizer
to convert them into a format suitable for consumption by PyTorch models.
"""

# Standard library imports
import logging
from pathlib import Path
from typing import Dict, List, Optional

# External library imports
import torch
from torch.utils.data import Dataset

# Internal project imports
from lspo_3d.data.tokenizer import CSGTokenizer
from lspo_3d import config
from lspo_3d import utils

# Configure logger
logger = logging.getLogger(__name__)


class CSGDataset(Dataset):
    """
    A PyTorch Dataset to handle a corpus of Constructive Solid Geometry (CSG)
    scripts written in OpenSCAD.

    This class scans a given directory for `.scad` files, reads them, and uses
    a provided CSGTokenizer to convert the script text into tokenized integer
    sequences. It is designed to be used with a PyTorch DataLoader for training
    generative or encoder models.

    Attributes:
        data_dir (str): The root directory containing the `.scad` files.
        tokenizer (CSGTokenizer): An initialized tokenizer object for converting
            scripts to tokens.
        max_seq_length (int): The maximum length of a token sequence. Scripts will
            be padded or truncated to this length.
        file_paths (List[Path]): A list of `pathlib.Path` objects pointing to
            all the discovered `.scad` files in the data directory.
        pad_token_id (int): The token ID used for padding sequences.
    """

    def __init__(
        self,
        data_dir: str,
        tokenizer: CSGTokenizer,
        max_seq_length: Optional[int] = None,
    ):
        """
        Initializes the CSGDataset.

        Args:
            data_dir (str): The path to the directory containing the dataset
                of `.scad` files. The directory will be searched recursively.
            tokenizer (CSGTokenizer): An instance of the CSGTokenizer to be
                used for encoding the scripts.
            max_seq_length (Optional[int]): The maximum sequence length for
                the tokenized output. If None, it will default to the value
                in the project configuration.
        """
        self.data_dir: str = data_dir
        self.tokenizer: CSGTokenizer = tokenizer
        self.max_seq_length: int = (
            max_seq_length if max_seq_length is not None else config.MAX_SEQ_LENGTH
        )
        
        pad_id = self.tokenizer.get_pad_token_id()
        if pad_id is None:
            raise ValueError("The provided tokenizer must have a defined PAD token.")
        self.pad_token_id: int = pad_id

        self.file_paths: List[Path] = self._load_file_paths()
        if not self.file_paths:
            logger.warning(f"No '.scad' files were found in the directory: {data_dir}")

    def _load_file_paths(self) -> List[Path]:
        """
        Scans the data directory recursively and collects paths to all `.scad` files.

        Returns:
            List[Path]: A sorted list of `pathlib.Path` objects for each `.scad`
                        file found. Sorting ensures consistent dataset ordering.
        
        Raises:
            FileNotFoundError: If the specified `data_dir` does not exist or is not a directory.
        """
        path = Path(self.data_dir)
        if not path.is_dir():
            raise FileNotFoundError(f"Data directory not found or not a directory at: {self.data_dir}")

        # Use rglob to find all .scad files recursively and sort for reproducibility
        file_paths = sorted(list(path.rglob("*.scad")))
        
        logger.info(f"Found {len(file_paths)} .scad files in {self.data_dir}")
        return file_paths

    def __len__(self) -> int:
        """
        Returns the total number of samples (files) in the dataset.

        Returns:
            int: The total count of `.scad` files found in the data directory.
        """
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Retrieves, processes, and tokenizes a single `.scad` file from the dataset.

        This method performs the following steps:
        1. Gets the file path for the given index.
        2. Reads the raw text content from the `.scad` file.
        3. Encodes the text into a sequence of token IDs using the tokenizer.
        4. Truncates or pads the sequence to `self.max_seq_length`.
        5. Creates an attention mask for the sequence.
        6. Converts the token IDs and attention mask into PyTorch tensors.

        Args:
            idx (int): The index of the data sample to retrieve.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the processed data,
                with keys 'input_ids' and 'attention_mask'.
        """
        file_path = self.file_paths[idx]

        try:
            with file_path.open("r", encoding="utf-8") as f:
                script_text = f.read()
        except Exception as e:
            logger.error(f"Error reading or processing file {file_path}: {e}")
            # Depending on the strategy, you might return a dummy sample,
            # skip the index, or raise an exception. Here we raise it.
            raise IOError(f"Could not read file {file_path}") from e

        # 3. Tokenize the script
        token_ids = self.tokenizer.encode(script_text)

        # 4. Pad/Truncate the sequence
        seq_len = len(token_ids)
        
        if seq_len > self.max_seq_length:
            # Truncate the sequence from the end
            input_ids = token_ids[:self.max_seq_length]
            attention_mask = [1] * self.max_seq_length
        else:
            # Pad the sequence
            padding_needed = self.max_seq_length - seq_len
            input_ids = token_ids + [self.pad_token_id] * padding_needed
            attention_mask = [1] * seq_len + [0] * padding_needed

        # 5. Convert to tensor
        input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
        attention_mask_tensor = torch.tensor(attention_mask, dtype=torch.long)

        # 6. Return as a dictionary
        return {
            "input_ids": input_ids_tensor,
            "attention_mask": attention_mask_tensor,
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for the DataLoader.

    This function takes a list of dataset samples (dictionaries) and stacks them
    into a single batch tensor for each key. It is particularly useful if
    __getitem__ does not perform padding and sequence lengths are variable.
    Even with padding in __getitem__, this provides a standard way to form a batch.

    Args:
        batch (List[Dict[str, torch.Tensor]]): A list of dictionary samples,
            where each dictionary is an output of CSGDataset.__getitem__.

    Returns:
        Dict[str, torch.Tensor]: A dictionary where each value is a tensor
            representing the batched data, e.g., 'input_ids' is a tensor of
            shape (batch_size, sequence_length).
    """
    if not batch:
        return {}

    # 1. Extract 'input_ids' from each item in the batch and stack them.
    input_ids = torch.stack([item["input_ids"] for item in batch])
    
    # 2. Extract 'attention_mask' from each item and stack them.
    attention_mask = torch.stack([item["attention_mask"] for item in batch])

    # 3. Return the new dictionary of batched tensors.
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }