# -*- coding: utf-8 -*-
"""
Implements the GPT-2 style Transformer Decoder model for CSG sequence generation.

This module defines the CSGGenerator, a Transformer-based model responsible for
generating sequences of OpenSCAD tokens conditioned on a sequence of abstract
design motif IDs.
"""

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from src.lspo_3d import config


class CSGGenerator(nn.Module):
    """
    A GPT-2 style Transformer Decoder to generate CSG scripts from motif IDs.

    This model is conditioned on a sequence of motif IDs and autoregressively
    generates a sequence of CSG tokens. It uses a causal self-attention
    mechanism to ensure that the prediction for a token at a given position
    only depends on the preceding tokens.

    Attributes:
        vocab_size (int): The size of the vocabulary for CSG tokens.
        motif_vocab_size (int): The number of unique design motifs.
        embed_dim (int): The dimensionality of the embedding layers.
        num_heads (int): The number of attention heads in the Transformer.
        num_layers (int): The number of Transformer decoder layers.
        max_seq_len (int): The maximum length of sequences the model can handle.
        dropout (float): The dropout rate used in the model.
        token_embedding (nn.Embedding): Embedding layer for CSG tokens.
        motif_embedding (nn.Embedding): Embedding layer for motif IDs.
        positional_embedding (nn.Embedding): Embedding for token positions.
        transformer_encoder (nn.TransformerEncoder): The stack of Transformer layers.
            Note: An Encoder with a causal mask is used to implement a decoder-only
            architecture like GPT-2.
        output_head (nn.Linear): Final linear layer to project to vocab size.
        dropout_layer (nn.Dropout): Dropout layer.
    """

    def __init__(
        self,
        vocab_size: int = config.VOCAB_SIZE,
        motif_vocab_size: int = config.NUM_MOTIFS,
        embed_dim: int = config.GENERATOR_EMBEDDING_DIM,
        num_heads: int = config.GENERATOR_NUM_HEADS,
        num_layers: int = config.GENERATOR_NUM_LAYERS,
        max_seq_len: int = config.MAX_SEQUENCE_LENGTH,
        dropout: float = config.GENERATOR_DROPOUT,
    ) -> None:
        """
        Initializes the CSGGenerator model.

        Args:
            vocab_size (int): The size of the vocabulary for CSG tokens.
            motif_vocab_size (int): The number of unique design motifs (k).
            embed_dim (int): The dimensionality of the embedding layers.
            num_heads (int): The number of attention heads in the Transformer.
            num_layers (int): The number of Transformer decoder layers.
            max_seq_len (int): The maximum length of sequences for positional encoding.
            dropout (float): The dropout rate.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.motif_vocab_size = motif_vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.dropout = dropout

        # Embedding layers
        self.token_embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        self.motif_embedding = nn.Embedding(self.motif_vocab_size, self.embed_dim)
        self.positional_embedding = nn.Embedding(self.max_seq_len, self.embed_dim)

        # Transformer layers (using Encoder with causal mask for decoder-only style)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            dim_feedforward=self.embed_dim * 4,
            dropout=self.dropout,
            batch_first=True,
            activation="gelu"
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers
        )

        # Output projection layer
        self.output_head = nn.Linear(self.embed_dim, self.vocab_size)

        # Dropout layer
        self.dropout_layer = nn.Dropout(self.dropout)

    def _generate_causal_mask(self, size: int, device: torch.device) -> Tensor:
        """
        Generates a causal mask for self-attention.

        This mask prevents the model from attending to future tokens during training.

        Args:
            size (int): The sequence length.
            device (torch.device): The device to create the mask on.

        Returns:
            Tensor: A square mask of shape (size, size) with upper-triangular
                    elements set to -inf and others to 0.0.
        """
        mask = torch.triu(torch.ones(size, size, device=device, dtype=torch.float), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(self, token_ids: Tensor, motif_ids: Tensor) -> Tensor:
        """
        Performs the forward pass of the CSGGenerator.

        Args:
            token_ids (Tensor): A tensor of token IDs representing the input sequence.
                                Shape: (batch_size, seq_len).
            motif_ids (Tensor): A tensor of motif IDs to condition the generation.
                                Shape: (batch_size, num_motifs).

        Returns:
            Tensor: A tensor of logits over the vocabulary for each position in
                    the sequence. Shape: (batch_size, seq_len, vocab_size).
        """
        batch_size, seq_len = token_ids.shape
        device = token_ids.device

        # 1. Generate position IDs and causal attention mask
        if seq_len > self.max_seq_len:
            raise ValueError(f"Input sequence length ({seq_len}) exceeds model's max sequence length ({self.max_seq_len})")
        position_ids = torch.arange(0, seq_len, device=device).unsqueeze(0)
        causal_mask = self._generate_causal_mask(seq_len, device)

        # 2. Get embeddings
        token_embeds = self.token_embedding(token_ids)  # (B, S, E)
        pos_embeds = self.positional_embedding(position_ids)  # (1, S, E) -> broadcasts to (B, S, E)
        motif_embeds = self.motif_embedding(motif_ids) # (B, M, E)

        # 3. Combine motif embeddings into a single context vector and add to inputs
        # A simple approach is to average the motif embeddings.
        motif_context = motif_embeds.mean(dim=1, keepdim=True)  # (B, 1, E) -> broadcasts to (B, S, E)
        
        # Combine embeddings: token + positional + motif context
        input_embeds = token_embeds + pos_embeds + motif_context
        input_embeds = self.dropout_layer(input_embeds)
        
        # 4. Pass through Transformer layers
        # The causal mask ensures that attention is only paid to previous positions.
        transformer_output = self.transformer_encoder(
            src=input_embeds,
            mask=causal_mask
        )

        # 5. Project to vocabulary space
        logits = self.output_head(transformer_output)

        return logits

    @torch.no_grad()
    def generate(
        self,
        motif_ids: Tensor,
        start_token_id: int,
        end_token_id: int,
        pad_token_id: int,
        max_len: int = config.MAX_SEQUENCE_LENGTH,
        temperature: float = 1.0,
        top_k: Optional[int] = None
    ) -> Tensor:
        """
        Autoregressively generates a sequence of tokens given motif IDs.

        This method performs inference by repeatedly calling the forward pass,
        sampling the next token, and appending it to the input sequence until
        the end token is generated or max_len is reached.

        Args:
            motif_ids (Tensor): The conditioning motif IDs.
                                Shape: (batch_size, num_motifs).
            start_token_id (int): The token ID to begin generation with.
            end_token_id (int): The token ID that signals the end of generation.
            pad_token_id (int): The token ID for padding. Used for finished sequences.
            max_len (int): The maximum length of the generated sequence.
            temperature (float): The temperature for sampling. Higher values make
                                 the output more random.
            top_k (Optional[int]): If set, filters the logits to the top k
                                   most likely tokens before sampling.

        Returns:
            Tensor: The generated sequence of token IDs for each item in the batch.
                    Shape: (batch_size, generated_len).
        """
        self.eval()  # Set the model to evaluation mode
        
        batch_size = motif_ids.shape[0]
        device = motif_ids.device

        # 1. Initialize the sequence with the start token for each batch item.
        generated_seq = torch.full((batch_size, 1), start_token_id, dtype=torch.long, device=device)
        
        # Keep track of which sequences have finished generation
        is_finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # 2. Loop for max_len steps:
        for _ in range(max_len - 1):
            # a. Call forward() with the current sequence and motif_ids.
            outputs = self.forward(token_ids=generated_seq, motif_ids=motif_ids)
            
            # b. Get logits for the last token position.
            last_token_logits = outputs[:, -1, :] # (batch_size, vocab_size)

            # c. Apply temperature scaling.
            if temperature != 1.0:
                last_token_logits = last_token_logits / temperature

            # d. Apply top-k filtering if specified.
            if top_k is not None:
                v, _ = torch.topk(last_token_logits, min(top_k, last_token_logits.size(-1)))
                last_token_logits[last_token_logits < v[:, [-1]]] = -float('inf')

            # e. Apply softmax to get probabilities.
            probs = torch.nn.functional.softmax(last_token_logits, dim=-1)
            
            # f. Sample the next token using torch.multinomial.
            next_token = torch.multinomial(probs, num_samples=1) # (batch_size, 1)

            # Do not change tokens for sequences that have already finished
            next_token[is_finished] = pad_token_id

            # g. Append the new token to the sequence.
            generated_seq = torch.cat((generated_seq, next_token), dim=1)

            # h. Check if any new sequences have generated an end token.
            is_finished = is_finished | (next_token.squeeze() == end_token_id)
            
            # If all sequences in the batch are finished, we can stop.
            if is_finished.all():
                break

        return generated_seq