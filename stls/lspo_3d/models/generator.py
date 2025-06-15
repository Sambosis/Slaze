# -*- coding: utf-8 -*-
"""
Defines the sequence-to-sequence model for translating design motifs to CadQuery scripts.

This module contains the CadQueryGenerator class, which wraps a pre-trained
transformer model (e.g., T5 or GPT-2) from the Hugging Face library. Its primary
function is to take a high-level design prompt and a sequence of abstract
motif IDs and generate a concrete, executable Python script using the CadQuery
library. It also includes methods for fine-tuning, saving, and loading the model.
"""

from __future__ import annotations
from rich import print as rr
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainingArguments
)
from typing import List, Dict, Any, Optional
from pathlib import Path

# The `trl` library is required for DPO. A user would need to install it.
# We import it here but make its usage conditional to avoid a hard dependency.
try:
    from trl import DPOTrainer
    _has_trl = True
except ImportError:
    _has_trl = False


def _format_input_for_generation(prompt: str, motif_ids: List[int]) -> str:
    """
    Helper function to create a standardized input string for the model.

    This ensures that the model always receives inputs in a consistent format,
    separating the natural language prompt from the abstract motif sequence.
    This structured prompting helps the model understand the task.

    Args:
        prompt (str): The natural language design prompt.
        motif_ids (List[int]): The sequence of motif IDs.

    Returns:
        str: A formatted string ready for tokenization.
    """
    motif_str = " ".join(map(str, motif_ids))
    # A clear, structured format is crucial for the model to learn.
    # The trailing instruction tells the model what to generate next.
    return (
        f"DESIGN PROMPT: {prompt.strip()} "
        f"| DESIGN MOTIFS: {motif_str} "
        f"| CADQUERY SCRIPT:"
    )


class CadQueryGenerator(torch.nn.Module):
    """
    A sequence-to-sequence model to generate CadQuery scripts from motif sequences.

    This class encapsulates a transformer model and its tokenizer to perform the
    translation from a conceptual design (prompt + motifs) to a practical
    CadQuery script.

    Attributes:
        model (PreTrainedModel): The underlying pre-trained transformer model
            (e.g., GPT2LMHeadModel).
        tokenizer (PreTrainedTokenizerBase): The tokenizer corresponding to the
            transformer model.
        device (torch.device): The device (CPU or CUDA) on which the model is loaded.
    """



    def __init__(
        self,
        model_name_or_path: str | Path,
        device: Optional[str] = None
    ):
        """
        Initializes the CadQueryGenerator.

        Args:
            model_name_or_path (str | Path): The identifier of the pre-trained
                model from Hugging Face Hub (e.g., "gpt2") or a path to a
                local directory containing the model and tokenizer files.
            device (Optional[str]): The device to run the model on, e.g., "cpu"
                or "cuda". If None, it will be automatically detected.
        """
        super().__init__()
        # 1. Determine device (cuda if available, else cpu).
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 2. Load the pre-trained model and tokenizer.
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_name_or_path)

        # 3. Set special tokens if necessary.
        # For causal LMs, pad_token is often not set. We set it to eos_token
        # to enable correct padding for batch operations.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            # Also update model config if pad_token_id is not set
            self.model.config.pad_token_id = self.model.config.eos_token_id

        # 4. Move the model to the selected device and set to eval mode.
        self.model.to(self.device).eval()
        print(f"CadQueryGenerator initialized on device: '{self.device}'")

    def generate(
        self,
        design_prompt: str,
        motif_ids: List[int],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        num_beams: int = 1
    ) -> str:
        """
        Generates a CadQuery script from a prompt and a sequence of motif IDs.

        This is the core inference method. It formats the input, tokenizes it,
        feeds it to the model, and decodes the output to produce a Python script.

        Args:
            design_prompt (str): The high-level design goal, e.g.,
                "Design a vertical stand for an iPhone 15".
            motif_ids (List[int]): A sequence of integers, where each integer
                is an ID representing a learned design motif.
            max_new_tokens (int): The maximum number of tokens to generate for the
                CadQuery script.
            temperature (float): Controls the randomness of the generation. Lower
                values make the output more deterministic. Only used if `num_beams` is 1.
            num_beams (int): The number of beams to use for beam search. `1` means
                greedy or sampling a single sequence.

        Returns:
            str: The generated CadQuery Python script as a string.
        """
        # 1. Format the input string.
        formatted_input = _format_input_for_generation(design_prompt, motif_ids)
        rr(f"[bold green]Formatted Input:[/bold green] {formatted_input}")
        # 2. Tokenize the input string and move to the model's device.
        inputs = self.tokenizer(formatted_input, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        prompt_input_len = input_ids.shape[1]

        # 3. Call `self.model.generate()` within a no_grad context for efficiency.
        with torch.no_grad():
            generation_kwargs = {
                "max_new_tokens": max_new_tokens,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "num_beams": num_beams,
            }
            # Temperature and sampling are typically used when not doing complex beam search.
            if num_beams == 1:
                generation_kwargs["do_sample"] = True
                generation_kwargs["temperature"] = temperature

            output_sequences = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generation_kwargs,
            )
        rr(f"[bold green]Output Sequences:[/bold green] {output_sequences}")
        # 4. Decode just the newly generated tokens back into a string.
        new_tokens = output_sequences[0, prompt_input_len:]
        generated_script = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        rr(f"[bold green]Generated Script:[/bold green] {generated_script}")
        # 5. Return the cleaned-up script.
        return generated_script.strip()

    def fine_tune_with_dpo(
        self,
        dpo_dataset: Any,
        training_args_dict: Dict[str, Any]
    ) -> None:
        """
        Fine-tunes the generator model using Direct Preference Optimization (DPO).

        This method refines the model based on pairs of preferred and rejected
        CadQuery scripts for given inputs, aligning it better with manufacturability
        and user preferences.

        Args:
            dpo_dataset (Any): A dataset object (e.g., from `datasets` library)
                containing 'prompt', 'chosen', and 'rejected' columns. This will
                require the `trl` library.
            training_args_dict (Dict[str, Any]): A dictionary of arguments to
                be passed to the `transformers.TrainingArguments` constructor.
        """
        if not _has_trl:
            raise ImportError(
                "DPO fine-tuning requires the `trl` library. "
                "Please install it with `pip install trl`."
            )

        # 1. Instantiate `transformers.TrainingArguments`.
        training_args = TrainingArguments(**training_args_dict)

        # 2. Instantiate the `trl.DPOTrainer`.
        # The model must be in training mode.
        self.model.train()
        dpo_trainer = DPOTrainer(
            model=self.model,
            args=training_args,
            train_dataset=dpo_dataset,
            tokenizer=self.tokenizer,
        )

        # 3. Call `trainer.train()` to start the fine-tuning process.
        print("Starting DPO fine-tuning...")
        dpo_trainer.train()
        print("DPO fine-tuning complete.")

        # 4. Save the fine-tuned model afterwards.
        self.model.eval() # Return to evaluation mode
        self.save_model(training_args.output_dir)

    def save_model(self, output_dir: str | Path) -> None:
        """
        Saves the model and tokenizer to a specified directory.

        Args:
            output_dir (str | Path): The directory where the model and
                tokenizer files will be saved.
        """
        # 1. Ensure the output directory exists.
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 2. Call `save_pretrained` for both model and tokenizer.
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        print(f"Model and tokenizer saved to '{output_path}'")

    @classmethod
    def load_model(cls, model_dir: str | Path, device: Optional[str] = None) -> CadQueryGenerator:
        """
        Loads a CadQueryGenerator model from a directory.

        This is a factory method to easily instantiate the class from a previously
        saved state.

        Args:
            model_dir (str | Path): The directory containing the saved model
                and tokenizer files.
            device (Optional[str]): The device to run the model on. If None,
                it will be auto-detected.

        Returns:
            CadQueryGenerator: A new instance of the class with the loaded weights.
        """
        # The constructor handles loading from a local path.
        return cls(model_name_or_path=model_dir, device=device)