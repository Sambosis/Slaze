# -*- coding: utf-8 -*-
"""
Inference script to generate 3D models from a user prompt.

This script loads the trained PPO Agent and the CAD Query Generator models.
It takes a natural language prompt from the user, uses the agent to predict an
optimal sequence of design motifs, translates this sequence into an executable
`cadquery` script using the generator, and finally executes the script to
produce a final .stl 3D model file.
"""

import argparse
import torch
from pathlib import Path
from typing import List, Tuple, Dict, Any

# A dummy text embedder. In a real application, this would be a proper
# model like one from sentence-transformers, consistent with what was used
# during training.
from sentence_transformers import SentenceTransformer

# Internal project imports
from src.lspo_3d.models.agent import AgentPolicy
from lspo_3d.models.generator import CadQueryGenerator
from src.lspo_3d.oracles.csg_executor import execute_cad_script

# --- Constants for Generation ---
# These should match the values used during training.
# Using placeholders as the exact values aren't specified.
MAX_MOTIF_STEPS = 15
EOS_MOTIF_ID = 0  # Assuming motif '0' is the end-of-sequence token.
PROMPT_EMBEDDING_DIM = 768 # Example dimension, should match agent's state_dim part
STATE_DIM = PROMPT_EMBEDDING_DIM + MAX_MOTIF_STEPS # Example, must match training

# Global cache for the sentence transformer model to avoid reloading.
_text_embedder = None

def _get_prompt_embedding(prompt: str, device: str) -> torch.Tensor:
    """
    Generates a fixed-size embedding for the given text prompt.

    NOTE: This is a placeholder implementation. A real system would use the
    exact same text embedding model (e.g., from sentence-transformers) that was
    used to train the agent in `train_agent.py`.

    Args:
        prompt (str): The input text prompt.
        device (str): The device ('cpu' or 'cuda') to place the tensor on.

    Returns:
        torch.Tensor: The prompt embedding tensor.
    """
    global _text_embedder
    if _text_embedder is None:
        # Load a standard sentence transformer model.
        # This model should ideally be specified in a config file.
        print("Loading text embedding model for the first time...")
        _text_embedder = SentenceTransformer('all-MiniLM-L6-v2', device=device)

    # The agent expects a batch dimension, so we add one and then remove it.
    embedding = _text_embedder.encode(
        prompt,
        convert_to_tensor=True,
        show_progress_bar=False
    ).to(device)
    
    # In a real scenario, we might need to pad/truncate to a fixed PROMPT_EMBEDDING_DIM
    # For this example, we assume the model output matches.
    # We'll assert this to catch mismatches.
    actual_dim = embedding.shape[0]
    if actual_dim != PROMPT_EMBEDDING_DIM:
        print(f"Warning: Text embedder output dim ({actual_dim}) does not match "
              f"expected dim ({PROMPT_EMBEDDING_DIM}). Truncating/Padding.")
        if actual_dim > PROMPT_EMBEDDING_DIM:
            embedding = embedding[:PROMPT_EMBEDDING_DIM]
        else:
            padding = torch.zeros(PROMPT_EMBEDDING_DIM - actual_dim, device=device)
            embedding = torch.cat((embedding, padding))
            
    return embedding


def _create_agent_state(
    prompt_embedding: torch.Tensor,
    motif_sequence: List[int],
    max_len: int,
    device: str
) -> torch.Tensor:
    """
    Creates a state tensor for the agent by combining the prompt and sequence.

    This function must be consistent with the state representation used during
    training in `DesignEnvironment`. It flattens the dictionary-like state
    into a single vector.

    Args:
        prompt_embedding (torch.Tensor): The embedding of the design prompt.
        motif_sequence (List[int]): The current sequence of generated motifs.
        max_len (int): The maximum sequence length for padding.
        device (str): The device to place the tensor on.

    Returns:
        torch.Tensor: A single state vector for the agent.
    """
    # Pad the current sequence to the maximum length.
    padded_sequence = motif_sequence + [EOS_MOTIF_ID] * (max_len - len(motif_sequence))
    sequence_tensor = torch.tensor(padded_sequence, dtype=torch.float32, device=device)

    # Concatenate prompt embedding and sequence tensor to form the final state.
    state = torch.cat((prompt_embedding, sequence_tensor)).unsqueeze(0)  # Add batch dim
    return state


def load_models(
    agent_path: str,
    generator_path: str,
    device: str
) -> Tuple[AgentPolicy, CadQueryGenerator]:
    """
    Loads the trained agent and generator models from specified paths.

    Args:
        agent_path (str): The file path to the saved agent checkpoint file.
                          This file should contain the model's state_dict and hyperparameters.
        generator_path (str): The directory path for the saved CadQueryGenerator model.
        device (str): The device to load the models onto ('cpu' or 'cuda').

    Returns:
        Tuple[AgentPolicy, CadQueryGenerator]: A tuple containing the loaded
                                               agent and generator models.
    """
    # --- Load Generator Model ---
    # The CadQueryGenerator.load_model classmethod handles its own loading.
    generator = CadQueryGenerator.load_model(generator_path)
    generator.model.to(device)
    generator.eval()

    # --- Load Agent Model ---
    # We assume the agent is saved in a checkpoint dictionary.
    checkpoint = torch.load(agent_path, map_location=device)

    # These hyperparameters are needed to initialize the model architecture.
    state_dim = checkpoint.get('state_dim', STATE_DIM)
    action_dim = checkpoint.get('action_dim')
    if action_dim is None:
        raise ValueError("Agent checkpoint must contain 'action_dim'.")

    agent = AgentPolicy(state_dim=state_dim, action_dim=action_dim)
    agent.load_state_dict(checkpoint['state_dict'])
    agent.to(device)
    agent.eval()

    return agent, generator


@torch.no_grad()
def generate_motif_sequence(
    prompt: str,
    agent: AgentPolicy,
    max_steps: int = MAX_MOTIF_STEPS,
    device: str = "cpu"
) -> List[int]:
    """
    Uses the trained agent to autoregressively generate a sequence of motif IDs.

    The generation process starts with the initial state (the prompt) and
    iteratively queries the agent for the next best action (motif) until an

    Args:
        prompt (str): The user's design prompt.
        agent (AgentPolicy): The loaded PPO agent model.
        max_steps (int): The maximum number of motifs to generate for the sequence.
        device (str): The device on which to perform inference.

    Returns:
        List[int]: A list of generated design motif IDs.
    """
    prompt_embedding = _get_prompt_embedding(prompt, device=device)
    motif_sequence = []

    for _ in range(max_steps):
        state = _create_agent_state(prompt_embedding, motif_sequence, max_steps, device)
        action, _, _ = agent.get_action_and_value(state)
        action_id = action.item()

        if action_id == EOS_MOTIF_ID and motif_sequence:
            break  # End generation if EOS token is produced (and sequence is not empty)

        motif_sequence.append(action_id)

    return motif_sequence


def generate_3d_model(
    prompt: str,
    agent_model_path: str,
    generator_model_path: str,
    output_stl_path: str,
    device: str
) -> None:
    """
    Orchestrates the full generation pipeline from prompt to .stl file.

    This function coordinates loading models, generating a motif sequence,
    translating it to a CadQuery script, and executing it to save the 3D model.

    Args:
        prompt (str): The user's design prompt (e.g., "Design a phone stand").
        agent_model_path (str): Path to the trained agent model checkpoint.
        generator_model_path (str): Path to the trained generator model directory.
        output_stl_path (str): The file path where the final .stl model will be saved.
        device (str): The device to run inference on ('cpu' or 'cuda').
    """
    print(f"Loading models from '{agent_model_path}' and '{generator_model_path}'...")
    agent, generator = load_models(agent_model_path, generator_model_path, device)
    print("Models loaded successfully.")

    print(f"Generating motif sequence for prompt: '{prompt}'...")
    motif_sequence = generate_motif_sequence(prompt, agent, device=device)
    if not motif_sequence:
        print("Model generated an empty motif sequence. Aborting.")
        return
    print(f"Generated motif sequence: {motif_sequence}")

    print("Translating motif sequence to CadQuery script...")
    cadquery_script = generator.generate(
        design_prompt=prompt,
        motif_ids=motif_sequence,
        max_new_tokens=1024  # Allow for longer scripts
    )
    print("Script generated successfully.")
    print("--- Generated Script ---")
    print(cadquery_script)
    print("------------------------")

    print(f"Executing script and saving model to '{output_stl_path}'...")
    output_path_obj = Path(output_stl_path)
    output_dir = str(output_path_obj.parent)
    output_filename = output_path_obj.stem # .stl is appended by the executor

    generated_file = execute_cad_script(
        script_string=cadquery_script,
        output_dir=output_dir,
        output_filename=output_filename
    )

    if generated_file:
        print(f"Successfully generated and saved 3D model to '{generated_file}'")
    else:
        print(f"Failed to generate 3D model. Check script for errors.")


def main() -> None:
    """
    Main function to parse command-line arguments and run the generation process.
    """
    parser = argparse.ArgumentParser(
        description="Generate a 3D model from a text prompt using a trained 3D-LSPO agent."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="The text prompt describing the desired 3D model."
    )
    parser.add_argument(
        "--agent-model-path",
        type=str,
        required=True,
        help="Path to the trained PPO agent model checkpoint file."
    )
    parser.add_argument(
        "--generator-model-path",
        type=str,
        required=True,
        help="Path to the trained CadQuery generator model directory."
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="File path to save the final generated .stl file."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the models on ('cuda' or 'cpu')."
    )

    args = parser.parse_args()

    # Ensure output directory exists
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)

    generate_3d_model(
        prompt=args.prompt,
        agent_model_path=args.agent_model_path,
        generator_model_path=args.generator_model_path,
        output_stl_path=args.output_path,
        device=args.device
    )


if __name__ == "__main__":
    main()