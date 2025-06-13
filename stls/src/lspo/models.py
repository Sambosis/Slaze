import torch
import torch.nn as nn
from torch.distributions import Categorical
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Optional

# The config must be imported to provide default values for model names.
from lspo.config import get_config

# Get the singleton config instance
_cfg = get_config()


class SequenceEmbedder:
    """
    A wrapper for the SentenceTransformer model to generate embeddings.

    This class handles loading a pre-trained sentence transformer model and
    provides a simple interface to embed sequences of text, such as
    cadquery command sequences, into fixed-size vectors.
    """

    def __init__(self, model_name: str = _cfg.embedding.model_name, device: Optional[str] = None):
        """
        Initializes the SequenceEmbedder.

        Args:
            model_name (str): The name or path of the sentence-transformer model
                              to load from the Hugging Face hub.
            device (Optional[str]): The device to run the model on (e.g., 'cpu', 'cuda').
                                    If None, it will be automatically detected.
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model: SentenceTransformer = SentenceTransformer(model_name, device=self.device)
        print(f"SequenceEmbedder initialized on device: {self.device}")

    def embed(self, sequences: List[str]) -> torch.Tensor:
        """
        Encodes a list of text sequences into embeddings.

        Args:
            sequences (List[str]): A list of strings to be embedded.

        Returns:
            torch.Tensor: A tensor of shape (num_sequences, embedding_dim)
                          containing the embeddings on the specified device.
        """
        # The encode method handles batching internally.
        # `convert_to_tensor=True` ensures the output is a PyTorch tensor.
        embeddings = self.model.encode(
            sequences,
            convert_to_tensor=True,
            device=self.device,
            show_progress_bar=False
        )
        return embeddings


class Generator(nn.Module):
    """
    The T5-based generator model.

    This model takes a sequence of design motifs (and potentially a text prompt)
    and generates a corresponding cadquery Python script. It is built upon the
    T5ForConditionalGeneration model from the transformers library.
    """

    def __init__(self, model_name: str = _cfg.generator.model_name, device: Optional[str] = None):
        """
        Initializes the Generator model and its tokenizer.

        Args:
            model_name (str): The name of the pre-trained T5 model to use. Can be a
                              path to a local checkpoint directory.
            device (Optional[str]): The device to run the model on (e.g., 'cpu', 'cuda').
                                    If None, it will be automatically detected.
        """
        super().__init__()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model.to(self.device)
        print(f"Generator model '{model_name}' initialized on device: {self.device}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a forward pass through the T5 model.

        This is primarily used during training (e.g., supervised fine-tuning or DPO)
        to calculate the loss.

        Args:
            input_ids (torch.Tensor): The tokenized input sequences.
            attention_mask (torch.Tensor): The attention mask for the input sequences.
            labels (Optional[torch.Tensor]): The tokenized target sequences (ground truth scripts).
                                             If provided, the model calculates and returns loss.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the loss (if labels are provided, else 0)
                                               and the logits.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        # Handle case where labels are not provided (e.g., during inference with DPO)
        loss = outputs.loss if outputs.loss is not None else torch.tensor(0.0, device=self.device)
        return loss, outputs.logits

    def generate_script(self, prompt: str, max_length: int = _cfg.generator.max_seq_length, num_beams: int = 5) -> str:
        """
        Generates a cadquery script from a given prompt string.

        The prompt should contain the necessary information for the model, such as
        the text description and the sequence of abstract design motifs.

        Args:
            prompt (str): The input text to the generator.
            max_length (int): The maximum length of the generated script in tokens.
            num_beams (int): The number of beams for beam search decoding.

        Returns:
            str: The generated cadquery Python script as a string.
        """
        # Tokenize the input prompt and move tensors to the correct device
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=max_length,
            truncation=True
        ).to(self.device)

        # Generate output token sequences
        output_sequences = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True  # Stop when a full sequence is generated
        )

        # Decode the generated tokens into a string
        generated_script = self.tokenizer.decode(
            output_sequences[0],
            skip_special_tokens=True
        )

        return generated_script

    def save_checkpoint(self, checkpoint_path: str) -> None:
        """
        Saves the model and tokenizer to a specified directory.

        Args:
            checkpoint_path (str): The directory path where the model and
                                   tokenizer will be saved.
        """
        self.model.save_pretrained(checkpoint_path)
        self.tokenizer.save_pretrained(checkpoint_path)
        print(f"Generator checkpoint saved to {checkpoint_path}")

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, device: Optional[str] = None) -> "Generator":
        """
        Loads a Generator instance from a checkpoint directory.

        Args:
            checkpoint_path (str): The directory path from which to load the model.
            device (Optional[str]): The device to load the model onto.

        Returns:
            Generator: A new instance of the Generator loaded with pre-trained weights.
        """
        return cls(model_name=checkpoint_path, device=device)


class Actor(nn.Module):
    """
    The Actor network for the PPO agent.

    This network implements the policy, which maps a state representation to a
    probability distribution over the discrete action space (design motifs).
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = _cfg.agent.actor_hidden_dims):
        """
        Initializes the Actor network architecture.

        Args:
            state_dim (int): The dimensionality of the input state vector.
            action_dim (int): The number of possible discrete actions (design motifs).
            hidden_dims (List[int]): Dimensions of the hidden layers.
        """
        super().__init__()
        layers = []
        in_dim = state_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        
        layers.append(nn.Linear(in_dim, action_dim))
        layers.append(nn.Softmax(dim=-1))
        
        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> Categorical:
        """
        Performs a forward pass to get the action distribution for a given state.

        Args:
            state (torch.Tensor): The input state tensor.

        Returns:
            torch.distributions.Categorical: A categorical distribution over the actions,
                                             from which an action can be sampled.
        """
        action_probs = self.network(state)
        # Create a categorical distribution from the output probabilities
        distribution = Categorical(probs=action_probs)
        return distribution


class Critic(nn.Module):
    """
    The Critic network for the PPO agent.

    This network estimates the value (expected return) of a given state, V(s).
    It is used to compute the advantage function during PPO updates.
    """

    def __init__(self, state_dim: int, hidden_dims: List[int] = _cfg.agent.critic_hidden_dims):
        """
        Initializes the Critic network architecture.

        Args:
            state_dim (int): The dimensionality of the input state vector.
            hidden_dims (List[int]): Dimensions of the hidden layers.
        """
        super().__init__()
        layers = []
        in_dim = state_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, 1)) # Outputs a single scalar value

        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass to estimate the value of a given state.

        Args:
            state (torch.Tensor): The input state tensor.

        Returns:
            torch.Tensor: A tensor containing the estimated value of the state.
        """
        # Squeeze to remove the last dimension, making it a scalar tensor
        return self.network(state)