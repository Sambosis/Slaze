# -*- coding: utf-8 -*-
"""
Main entry point for the 3D-LSPO application.

This script handles command-line arguments to orchestrate the primary modes
of operation: training a new model or running inference with a pre-trained
model to generate a 3D object from a text prompt.
"""

import argparse
import sys
from typing import Optional

import torch

from lspo import config
from lspo import trainer


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments for the application.

    Sets up two main sub-commands: 'train' and 'inference', each with its
    own set of specific arguments.

    Returns:
        argparse.Namespace: An object containing the parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="3D-LSPO: Latent Strategy Optimization for 3D Printable Models."
    )
    
    subparsers = parser.add_subparsers(dest='command', required=True,
                                       help='Available commands')

    # --- Training Parser ---
    parser_train = subparsers.add_parser(
        'train', help='Start the full training process for the LSPO models.'
    )
    parser_train.add_argument(
        '--config_path', type=str, default=None,
        help='Optional path to a custom configuration file.'
    )
    parser_train.add_argument(
        '--checkpoint', type=str, default=None,
        help='Optional path to a model checkpoint to resume training from.'
    )
    
    # --- Inference Parser ---
    parser_inference = subparsers.add_parser(
        'inference', help='Run inference using a pre-trained model checkpoint.'
    )
    parser_inference.add_argument(
        '--prompt', type=str, required=True,
        help='The high-level text prompt to generate a 3D model for.'
    )
    parser_inference.add_argument(
        '--checkpoint', type=str, required=True,
        help='Path to the pre-trained model checkpoint to use for generation.'
    )
    parser_inference.add_argument(
        '--output_path', type=str, default=None,
        help=(
            'Optional path to save the generated STL file. '
            'If not provided, a default path from the config will be used.'
        )
    )

    return parser.parse_args()


def run_training(args: argparse.Namespace) -> None:
    """
    Initializes and runs the model training process.

    This function orchestrates the setup of the training environment,
    loading of configurations, and initiation of the main training loop
    as defined in the `trainer` module.

    Args:
        args (argparse.Namespace): The parsed command-line arguments,
                                   expected to contain `config_path` and
                                   `checkpoint` for the training command.
    """
    print("--- Starting 3D-LSPO Training ---")

    # Load project configurations. The skeleton for config.py suggests get_config()
    # is the main entry point. A more advanced version could load from args.config_path.
    if args.config_path:
        print(f"Warning: Loading from custom config path '{args.config_path}' is not yet implemented. Using default config.")
    project_config = config.get_config()
    print(f"Loaded configuration for project: '{project_config.project_name}'")

    # Determine the device for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training will use device: {device}")

    # Instantiate the main Trainer class from the trainer module.
    # The LSPOTrainer is expected to handle the entire training lifecycle.
    lspo_trainer = trainer.LSPOTrainer(cfg=project_config, device=device)
    
    # If a checkpoint is provided, load the state for the models and optimizers.
    if args.checkpoint:
        print(f"Resuming training from checkpoint: {args.checkpoint}")
        lspo_trainer.load_checkpoint(args.checkpoint)
    else:
        print("Starting a new training session.")

    # Start the main training loop.
    lspo_trainer.train()
    
    print("--- Training Completed ---")


def run_inference(args: argparse.Namespace) -> None:
    """
    Runs the inference pipeline to generate a 3D model.

    This function loads a pre-trained model, processes the input text prompt,
    and executes the generation pipeline to produce a final STL file.

    Args:
        args (argparse.Namespace): The parsed command-line arguments,
                                   expected to contain `prompt`, `checkpoint`,
                                   and `output_path` for the inference command.
    """
    print(f"--- Running 3D-LSPO Inference for prompt: '{args.prompt}' ---")
    
    # Load project configurations.
    project_config = config.get_config()

    # The design suggests a dedicated class for inference within the trainer module.
    # We will attempt to import and use it, providing a clear error if it's not found.
    try:
        # pylint: disable=no-name-in-module
        from lspo.trainer import InferencePipeline
    except (ImportError, AttributeError):
        print("\nError: Could not find 'InferencePipeline' in 'lspo.trainer'.", file=sys.stderr)
        print("Inference mode requires this class to be implemented in trainer.py.", file=sys.stderr)
        sys.exit(1)

    # Instantiate the inference pipeline, which loads models from the checkpoint.
    print(f"Loading inference pipeline from checkpoint: {args.checkpoint}")
    inference_pipeline = InferencePipeline(
        checkpoint_path=args.checkpoint,
        config=project_config
    )

    # Execute the inference process with the user's prompt.
    print("Generating model...")
    generated_stl_path = inference_pipeline.generate(
        prompt=args.prompt,
        output_path=args.output_path
    )
    
    if generated_stl_path and generated_stl_path.exists():
        print("--- Inference Complete ---")
        print(f"Generated model saved to: {generated_stl_path}")
    else:
        print("--- Inference Failed ---")
        print("Model generation did not produce a valid STL file.")


def main() -> None:
    """
    Main function to drive the script's execution.

    Parses arguments and calls the appropriate handler function based on the
    selected command ('train' or 'inference').
    """
    args = parse_arguments()

    if args.command == 'train':
        run_training(args)
    elif args.command == 'inference':
        run_inference(args)
    else:
        # This case should not be reachable due to `required=True` in subparsers
        print(f"Error: Unknown command '{args.command}'", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()