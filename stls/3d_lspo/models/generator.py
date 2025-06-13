# 3d_lspo/models/generator.py

"""
Defines the sequence-to-sequence model that translates a sequence of high-level
design motif IDs into an executable cadquery script. This model is a core
component of the generation pipeline, responsible for the low-level implementation
of the agent's high-level strategy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Union

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    TrainingArguments,
)

# The trl library is essential for DPO training.
# Ensure it is installed: pip install trl
try:
    from trl import DPOTrainer
    from datasets import Dataset
    _TRL_AVAILABLE = True
except ImportError:
    DPOTrainer = None
    Dataset = None
    _TRL_AVAILABLE = False


class CadQueryGenerator:
    """
    A wrapper for a sequence-to-sequence transformer model for CadQuery code generation.

    This class encapsulates a pre-trained model (like T5) and its tokenizer to
    perform the translation from a structured input (a design prompt and a list
    of motif IDs) to a string of Python code using the cadquery library.

    Attributes:
        model (PreTrainedModel): The underlying transformer model for sequence-to-sequence tasks.
        tokenizer (PreTrainedTokenizer): The tokenizer corresponding to the model.
        device (torch.device): The device (CPU or CUDA) on which the model is loaded.
        model_name_or_path (Union[str, Path]): Stores the identifier used to load the model.
    """

    def __init__(
        self,
        model_name_or_path: Union[str, Path],
        device: str = "cpu"
    ):
        """
        Initializes the CadQueryGenerator by loading a pre-trained model and tokenizer.

        Args:
            model_name_or_path (Union[str, Path]): The identifier of a pre-trained
                model from the Hugging Face Hub (e.g., 't5-small') or a path to a
                local directory containing model files.
            device (str): The device to run the model on, e.g., "cpu" or "cuda".
        """
        self.device: torch.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_name_or_path = model_name_or_path

        self.model: PreTrainedModel = AutoModelForSeq2SeqLM.from_pretrained(
            model_name_or_path
        ).to(self.device)
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path
        )

    @classmethod
    def from_pretrained(
        cls,
        model_directory: Union[str, Path],
        device: str = "cpu"
    ) -> "CadQueryGenerator":
        """
        Class method to load a fine-tuned generator from a local directory.

        This serves as a factory function to instantiate the generator with
        a model that has already been saved using `save_pretrained`.

        Args:
            model_directory (Union[str, Path]): Path to the directory containing
                the saved model and tokenizer files.
            device (str): The device to load the model onto.

        Returns:
            CadQueryGenerator: An instance of the class with the loaded model.
        """
        return cls(model_name_or_path=model_directory, device=device)

    def save_pretrained(self, save_directory: Union[str, Path]) -> None:
        """
        Saves the model and tokenizer to a specified directory.

        This allows the state of the fine-tuned generator to be persisted
        for later use in inference or further training.

        Args:
            save_directory (Union[str, Path]): The path to the directory where
                the model and tokenizer will be saved.
        """
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"Model and tokenizer saved to {save_path}")

    def generate(
        self,
        prompt: str,
        motif_ids: List[int],
        max_length: int = 512,
        **generation_kwargs: Any
    ) -> str:
        """
        Generates a CadQuery script based on a prompt and a sequence of motifs.

        This method constructs a formatted input string from the prompt and motifs,
        tokenizes it, and feeds it to the underlying model to generate the
        corresponding CadQuery code.

        The expected input format for the underlying seq2seq model is a single
        string like:
        "prompt: <design_prompt> motifs: <id_1> <id_2> ... <id_n>"

        Args:
            prompt (str): The high-level design goal, e.g., "Design a phone stand".
            motif_ids (List[int]): A sequence of integer IDs representing the
                high-level design strategy chosen by the RL agent.
            max_length (int): The maximum length of the generated token sequence.
            **generation_kwargs (Any): Additional keyword arguments to be passed
                to the model's `generate` method (e.g., `num_beams`, `temperature`).

        Returns:
            str: The generated CadQuery Python script as a string.
        """
        self.model.eval()  # Ensure the model is in evaluation mode

        motif_ids_str = " ".join(map(str, motif_ids))
        input_text = f"prompt: {prompt} motifs: {motif_ids_str}"

        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(self.device)

        with torch.no_grad():
            output_sequences = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                **generation_kwargs
            )

        # Decode the generated token IDs to a string
        generated_script = self.tokenizer.decode(
            output_sequences[0],
            skip_special_tokens=True
        )

        return generated_script


def train_generator_with_dpo(
    generator: CadQueryGenerator,
    dataset: Sequence[Dict[str, Any]],
    output_dir: Union[str, Path],
    training_args: TrainingArguments,
    beta: float = 0.1,
) -> None:
    """
    Fine-tunes the CadQueryGenerator using Direct Preference Optimization (DPO).

    This function sets up and runs a DPO training loop. It requires a dataset
    of triplets: (prompt, chosen_completion, rejected_completion), where completions
    are CadQuery scripts. This aligns the generator model to prefer producing
    high-quality, manufacturable designs over flawed ones.

    Note: This function requires the `trl` and `datasets` libraries to be installed.
          `pip install trl datasets`

    Args:
        generator (CadQueryGenerator): The generator instance to be fine-tuned.
        dataset (Sequence[Dict[str, Any]]): A dataset where each item is a dictionary
            expected to contain 'prompt', 'chosen', and 'rejected' keys. 'prompt'
            is the input to the generator, 'chosen' is the preferred output script,
            and 'rejected' is the dispreferred output script.
        output_dir (Union[str, Path]): The directory where training checkpoints
            and the final model will be saved.
        training_args (TrainingArguments): Hugging Face TrainingArguments to
            configure the training process (learning rate, epochs, etc.).
        beta (float): The beta parameter for DPO, controlling the deviation
            from the reference model.
    """
    if not _TRL_AVAILABLE:
        raise ImportError(
            "The 'trl' and 'datasets' libraries are required for DPO training. "
            "Please install them using: pip install trl datasets"
        )
    
    if not dataset:
        print("Warning: DPO training dataset is empty. Skipping fine-tuning.")
        return

    # Ensure the output directory for TrainingArguments is set
    training_args.output_dir = str(output_dir)

    # Convert the list of dicts to a Hugging Face Dataset object
    hf_dataset = Dataset.from_list(list(dataset))

    # Initialize the DPOTrainer
    dpo_trainer = DPOTrainer(
        model=generator.model,
        ref_model=None,  # ref_model is optional and will be created from the model if None
        args=training_args,
        beta=beta,
        train_dataset=hf_dataset,
        tokenizer=generator.tokenizer,
    )

    print("Starting DPO fine-tuning...")
    # Run the training process
    dpo_trainer.train()
    print("DPO fine-tuning complete.")

    # Save the final fine-tuned model
    print(f"Saving DPO-tuned model to {output_dir}...")
    dpo_trainer.save_model(output_dir)
    print("Model saved successfully.")