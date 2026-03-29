import os
import glob
import shutil
import random
import re
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Union

import config

def save_model(
    agent: nn.Module, 
    filepath: str, 
    episode: int, 
    optimizer: Optional[torch.optim.Optimizer] = None, 
    **kwargs
) -> None:
    """
    Saves the agent's state dictionary, optimizer state, and metadata to a checkpoint file.

    Args:
        agent: The PyTorch model to save.
        filepath: The full path where the checkpoint will be written.
        episode: The current episode number.
        optimizer: Optional optimizer to save state for resuming training.
        **kwargs: Any additional metadata to store.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    checkpoint = {
        'model_state_dict': agent.state_dict(),
        'episode': episode,
        **kwargs
    }

    if optimizer:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

    torch.save(checkpoint, filepath)


def load_model(
    agent: nn.Module, 
    filepath: str, 
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Union[str, torch.device] = 'cpu'
) -> Dict[str, Any]:
    """
    Loads a checkpoint into the provided agent and optional optimizer.

    Args:
        agent: The model instance to load weights into.
        filepath: Path to the checkpoint file.
        optimizer: If provided, loads optimizer state.
        device: Device to map the checkpoint to (e.g., 'cpu', 'cuda').

    Returns:
        Dict: The full checkpoint dictionary containing metadata (e.g., 'episode').

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")

    # Load checkpoint with device mapping
    checkpoint = torch.load(filepath, map_location=device)

    # Load model weights
    agent.load_state_dict(checkpoint['model_state_dict'])
    agent.to(device)

    # Load optimizer state if requested and present
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint


class HallOfFame:
    """
    Manages the storage and retrieval of past model weights (historical opponents).
    
    This class maintains a directory of model checkpoints, enforces