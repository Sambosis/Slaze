# -*- coding: utf-8 -*-
"""
Implements the Transformer Encoder model for creating vector embeddings.

This model takes sequences of tokens representing Constructive Solid Geometry (CSG)
scripts and maps them into a high-dimensional latent space. These embeddings
are later used for clustering to discover abstract "design motifs".
"""

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from lspo_3d import config


class PositionalEncoding(nn.Module):
    """
    Injects positional information into the input embeddings.

    This module adds sinusoidal positional encodings to the token embeddings
    at the beginning of the encoder. This is crucial as the Transformer
    architecture itself does not have a notion of sequence order.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """
        Initializes the PositionalEncoding module.

        Args:
            d_model (int): The dimensionality of the embeddings (and the model).
            dropout (float): The dropout probability.
            max_len (int): The maximum possible length of a sequence.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Applies positional encoding to the input tensor.

        Args:
            x (Tensor): The input tensor of token embeddings.
                        Shape: (seq_len, batch_size, embedding_dim)

        Returns:
            Tensor: The input tensor with positional information added.
                    Shape: (seq_len, batch_size, embedding_dim)
        """
        # Add positional encoding. self.pe will be sliced to match sequence length
        # and broadcasted across the batch dimension.
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)



class CSGEncoder(nn.Module):
    """
    A Transformer-based encoder for CSG script token sequences.

    This class defines the main model architecture responsible for generating
    latent vector representations of OpenSCAD scripts. It is composed of an
    embedding layer, positional encoding, and a stack of Transformer encoder layers.

    Attributes:
        model_type (str): The type of the model, e.g., 'Transformer'.
        src_mask (Optional[Tensor]): A mask to prevent the model from attending to future tokens (not typically used in encoders, but can be for auto-encoding tasks).
        pos_encoder (PositionalEncoding): The positional encoding module.
        transformer_encoder (nn.TransformerEncoder): The stack of Transformer encoder layers.
        embedding (nn.Embedding): The layer that maps token IDs to vectors.
        d_model (int): The dimensionality of the model's embeddings.
    """

    model_type: str = 'Transformer'

    def __init__(self,
                 vocab_size: int = config.VOCAB_SIZE,
                 d_model: int = config.ENCODER_EMBEDDING_DIM,
                 nhead: int = config.ENCODER_NUM_HEADS,
                 d_hid: int = config.ENCODER_FF_DIM,
                 nlayers: int = config.ENCODER_NUM_LAYERS,
                 dropout: float = config.ENCODER_DROPOUT):
        """
        Initializes the CSGEncoder model.

        Args:
            vocab_size (int): The total number of unique tokens in the vocabulary.
            d_model (int): The dimensionality of the embeddings and the model.
            nhead (int): The number of attention heads in the multi-head attention mechanism.
            d_hid (int): The dimension of the feedforward network model.
            nlayers (int): The number of sub-encoder-layers in the encoder.
            dropout (float): The dropout probability.
        """
        super().__init__()

        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=False)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.src_mask: Optional[Tensor] = None

        self._init_weights()

    def _init_weights(self) -> None:
        """
        Initializes the weights of the model.

        This method applies a uniform distribution for the embedding layer
        and zeros for biases to ensure a good starting point for training.
        """
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        # Note: Linear layer biases inside TransformerEncoderLayer are initialized to zero by default.

    def forward(self, src: Tensor, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """
        Performs the forward pass of the encoder.

        This method takes a batch of tokenized sequences and produces their
        corresponding embeddings. The output can be pooled (e.g., by averaging or
        taking the first token's embedding) in a downstream task to get a
        fixed-size representation for the entire sequence.

        Args:
            src (Tensor): The input sequence of token IDs.
                        Shape: (seq_len, batch_size)
            src_key_padding_mask (Optional[Tensor]): A mask indicating which
                                                     elements are padding and should
                                                     be ignored by the attention
                                                     mechanism.
                                                     Shape: (batch_size, seq_len)

        Returns:
            Tensor: The output embeddings from the Transformer encoder.
                    Shape: (seq_len, batch_size, d_model)
        """
        # 1. Pass `src` through the embedding layer.
        src_embed = self.embedding(src)

        # 2. Scale the embeddings by sqrt(d_model).
        src_embed = src_embed * math.sqrt(self.d_model)

        # 3. Add positional encodings.
        src_pos_embed = self.pos_encoder(src_embed)

        # 4. Pass the result through the self.transformer_encoder with the padding mask.
        output = self.transformer_encoder(
            src=src_pos_embed,
            mask=self.src_mask,
            src_key_padding_mask=src_key_padding_mask
        )

        # 5. Return the final output tensor.
        return output