# src/lspo_3d/data/tokenizer.py

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Union

from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
from tokenizers.processors import TemplateProcessing

# Internal project imports
from src.lspo_3d import config
from lspo_3d import utils

logger = logging.getLogger(__name__)


class CSGTokenizer:
    """
    Handles the tokenization of OpenSCAD scripts.

    This class is a wrapper around the `tokenizers` library and is responsible
    for training a tokenizer on a corpus of `.scad` files, converting scripts
    to token IDs (encoding), and converting token IDs back to scripts (decoding).
    """

    def __init__(self, tokenizer_instance: Optional[Tokenizer] = None):
        """
        Initializes the CSGTokenizer.

        Can be initialized with a pre-configured Tokenizer instance or as an
        empty tokenizer to be trained later.

        Args:
            tokenizer_instance (Optional[Tokenizer]): An optional, pre-existing
                instance of a `tokenizers.Tokenizer`. If None, a new one
                will be created.
        """
        self.tokenizer: Tokenizer
        if tokenizer_instance:
            self.tokenizer = tokenizer_instance
        else:
            self.tokenizer = Tokenizer(models.BPE())
            self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
            self.tokenizer.decoder = decoders.ByteLevel()

        self.pad_token = config.PAD_TOKEN
        self.unk_token = config.UNK_TOKEN
        self.bos_token = config.BOS_TOKEN
        self.eos_token = config.EOS_TOKEN
        self.special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]


    def train(self, corpus_files: List[Union[str, Path]]) -> None:
        """
        Trains the tokenizer on a corpus of OpenSCAD files.

        This method configures and runs a trainer on the provided file paths
        to build the vocabulary and learn the merge rules. The tokenizer is
        updated in-place. After training, it configures a post-processor to
        automatically add Beginning-Of-Sequence (BOS) and End-Of-Sequence (EOS)
        tokens.

        Args:
            corpus_files (List[Union[str, Path]]): A list of paths to the `.scad`
                files that will be used for training the tokenizer.
        """
        logger.info(f"Starting tokenizer training with {len(corpus_files)} files.")
        trainer = trainers.BpeTrainer(
            vocab_size=config.TOKENIZER_VOCAB_SIZE,
            special_tokens=self.special_tokens,
            min_frequency=config.TOKENIZER_MIN_FREQUENCY
        )
        
        # The `train` method consumes a list of file paths as strings
        str_corpus_files = [str(f) for f in corpus_files]
        self.tokenizer.train(files=str_corpus_files, trainer=trainer)
        logger.info(f"Tokenizer training complete. Vocabulary size: {self.get_vocab_size()}")

        # Configure post-processing to add BOS/EOS tokens automatically
        bos_token_id = self.get_bos_token_id()
        eos_token_id = self.get_eos_token_id()

        if bos_token_id is None or eos_token_id is None:
            raise RuntimeError("BOS or EOS token ID not found in vocabulary after training.")

        self.tokenizer.post_processor = TemplateProcessing(
            single=f"{self.bos_token} $A {self.eos_token}",
            special_tokens=[(self.bos_token, bos_token_id), 
                            (self.eos_token, eos_token_id)],
        )
        logger.info("Tokenizer post-processor configured to add BOS/EOS tokens.")

    def encode(self, text: str) -> List[int]:
        """
        Encodes a raw OpenSCAD script string into a sequence of token IDs.

        Args:
            text (str): The raw OpenSCAD script to encode.

        Returns:
            List[int]: A list of integer token IDs representing the script.
        """
        encoding = self.tokenizer.encode(text)
        return encoding.ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decodes a sequence of token IDs back into a string.

        Args:
            token_ids (List[int]): The list of token IDs to decode.
            skip_special_tokens (bool): Whether to remove special tokens
                (like padding or BOS/EOS) from the output string.

        Returns:
            str: The reconstructed OpenSCAD script string.
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def save(self, file_path: Union[str, Path]) -> None:
        """
        Saves the tokenizer's state to a JSON file.

        Args:
            file_path (Union[str, Path]): The path where the tokenizer file
                will be saved. The directory will be created if it does not exist.
        """
        path = Path(file_path)
        # Ensure the target directory exists before saving.
        # This relies on a utility function defined in the project.
        utils.ensure_directory_exists(path.parent)
        self.tokenizer.save(str(path))
        logger.info(f"Tokenizer saved to {path}")

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> CSGTokenizer:
        """
        Loads a tokenizer from a saved JSON file.

        This is a factory method to create a CSGTokenizer instance from a
        pre-trained file.

        Args:
            file_path (Union[str, Path]): The path to the tokenizer JSON file.

        Returns:
            CSGTokenizer: A new instance of the tokenizer loaded from the file.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Tokenizer file not found at: {path}")
            
        tokenizer_instance = Tokenizer.from_file(str(path))
        logger.info(f"Tokenizer loaded from {path}")
        return cls(tokenizer_instance=tokenizer_instance)

    def get_vocab_size(self) -> int:
        """
        Returns the size of the tokenizer's vocabulary.

        Returns:
            int: The total number of tokens in the vocabulary.
        """
        return self.tokenizer.get_vocab_size()

    def get_pad_token_id(self) -> Optional[int]:
        """
        Returns the integer ID for the padding token.

        Returns:
            Optional[int]: The ID of the [PAD] token, or None if not found.
        """
        return self.tokenizer.token_to_id(self.pad_token)

    def get_bos_token_id(self) -> Optional[int]:
        """
        Returns the integer ID for the beginning-of-sequence token.

        Returns:
            Optional[int]: The ID of the [BOS] token, or None if not found.
        """
        return self.tokenizer.token_to_id(self.bos_token)
    
    def get_eos_token_id(self) -> Optional[int]:
        """
        Returns the integer ID for the end-of-sequence token.

        Returns:
            Optional[int]: The ID of the [EOS] token, or None if not found.
        """
        return self.tokenizer.token_to_id(self.eos_token)