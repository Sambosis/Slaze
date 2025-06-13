# C:\Users\Machine81\Slazy\repo\stls\3d_lspo\generate.py

"""
Inference script for 3D-LSPO.

This script takes a natural language design prompt and uses the trained
Agent and Generator models to produce a final, optimized 3D model in .stl
format. It orchestrates the process of generating a high-level design plan
(a sequence of motifs) with the agent, translating that plan into a concrete
CAD script with the generator, and finally executing the script to create the
3D file.
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import torch
# SentenceTransformer is a sensible choice for prompt encoding, as the agent
# needs a numerical representation of the text prompt. This was likely used
# during training.
from sentence_transformers import SentenceTransformer

# Internal project imports
from lspo_3d.models.agent import AgentPolicy
from lspo_3d.models.generator import CadQueryGenerator
from lspo_3d.oracles.csg_executor import execute_cad_script


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments for the generation script.

    Returns:
        argparse.Namespace: An object containing the parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate a 3D model from a text prompt using a trained 3D-LSPO agent."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="The natural language design prompt (e.g., 'Design a vertical stand for an iPhone 15')."
    )
    parser.add_argument(
        "--agent_model_path",
        type=Path,
        required=True,
        help="Path to the trained PPO agent model state_dict (.pth file)."
    )
    parser.add_argument(
        "--generator_model_path",
        type=Path,
        required=True,
        help="Path to the directory containing the trained CadQuery generator model."
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        required=True,
        help="Path to save the generated .stl file."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="The device to run inference on ('cuda' or 'cpu')."
    )
    parser.add_argument(
        "--max_motifs",
        type=int,
        default=10,
        help="The maximum number of design motifs the agent can select."
    )
    parser.add_argument(
        "--agent_state_dim_base",
        type=int,
        default=384,
        help="The base dimensionality of the agent's state space (embedding size for the prompt)."
    )
    parser.add_argument(
        "--agent_action_dim",
        type=int,
        default=100,
        help="The dimensionality of the agent's action space (number of unique motifs)."
    )
    return parser.parse_args()


def load_models(
    agent_path: Path,
    generator_path: Path,
    device: str,
    state_dim: int,
    action_dim: int
) -> Tuple[AgentPolicy, CadQueryGenerator]:
    """
    Loads the trained agent and generator models from disk.

    Args:
        agent_path (Path): The file path to the saved agent model's state dictionary.
        generator_path (Path): The directory path containing the saved generator model.
        device (str): The device to load the models onto (e.g., 'cpu' or 'cuda').
        state_dim (int): The total dimensionality of the agent's state space.
        action_dim (int): The dimensionality of the agent's action space.

    Returns:
        Tuple[AgentPolicy, CadQueryGenerator]: A tuple containing the initialized and
                                               loaded agent and generator models.
    """
    # Load AgentPolicy: initialize with architecture params, then load weights.
    agent_model = AgentPolicy(state_dim=state_dim, action_dim=action_dim)
    agent_model.load_state_dict(torch.load(agent_path, map_location=device))
    agent_model.to(device)
    agent_model.eval()
    print(f"AgentPolicy loaded from {agent_path}")

    # Load CadQueryGenerator: use the from_pretrained class method for HF models.
    generator_model = CadQueryGenerator.from_pretrained(generator_path, device=device)
    generator_model.model.eval()
    print(f"CadQueryGenerator loaded from {generator_path}")

    return agent_model, generator_model


def generate_motif_sequence(
    prompt: str,
    agent: AgentPolicy,
    text_encoder: SentenceTransformer,
    max_steps: int,
    device: str,
    pad_token_id: int = 0
) -> List[int]:
    """
    Uses the agent to generate a sequence of design motifs based on the prompt.

    This function simulates the agent's decision-making process, where the state
    is composed of the initial prompt and the sequence of motifs selected so far.
    The agent acts greedily, selecting the action (motif) with the highest
    probability at each step.

    Args:
        prompt (str): The initial design prompt.
        agent (AgentPolicy): The trained agent policy model.
        text_encoder (SentenceTransformer): The model to encode the text prompt.
        max_steps (int): The maximum number of motifs to generate.
        device (str): The device on which to perform inference.
        pad_token_id (int): The ID used for padding the motif sequence.

    Returns:
        List[int]: A list of integers, where each integer is a motif ID.
    """
    motif_sequence = []

    # Encode the initial prompt into a fixed-size embedding.
    prompt_embedding = torch.tensor(
        text_encoder.encode(prompt), dtype=torch.float32, device=device
    ).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        for _ in range(max_steps):
            # Construct the current state tensor.
            # State = prompt_embedding + padded motif_sequence
            current_sequence_padded = motif_sequence + [pad_token_id] * (max_steps - len(motif_sequence))
            sequence_tensor = torch.tensor(
                current_sequence_padded, dtype=torch.float32, device=device
            ).unsqueeze(0)  # Add batch dimension

            state = torch.cat((prompt_embedding, sequence_tensor), dim=1)

            # Get action from the agent. For inference, we get the distribution
            # from a forward pass and take the most likely action (argmax).
            dist, _ = agent(state)
            action = torch.argmax(dist.logits, dim=-1).item()

            # Append the chosen motif to the sequence for the next step.
            motif_sequence.append(action)

    return motif_sequence


def main() -> None:
    """
    Main execution function for the generation script.

    Orchestrates loading models, generating a design plan, translating it to
    CAD code, and executing it to produce the final .stl file.
    """
    args = parse_arguments()

    # Ensure output directory exists before saving files.
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading text encoder for prompt processing...")
    # This must match the model used during agent training for correct embeddings.
    text_encoder_model = 'all-MiniLM-L6-v2'  # A common choice.
    text_encoder = SentenceTransformer(text_encoder_model, device=args.device)

    # Validate that the provided base dimension matches the text encoder's output.
    encoder_dim = text_encoder.get_sentence_embedding_dimension()
    if encoder_dim != args.agent_state_dim_base:
        print(
            f"Warning: Text encoder embedding dim ({encoder_dim}) "
            f"does not match --agent_state_dim_base ({args.agent_state_dim_base}). "
            f"This may lead to unexpected behavior."
        )

    # The agent's full state dimension is the prompt embedding + the max motif sequence length.
    agent_state_dim = args.agent_state_dim_base + args.max_motifs

    print(f"Loading models from {args.agent_model_path} and {args.generator_model_path}...")
    try:
        agent, generator = load_models(
            agent_path=args.agent_model_path,
            generator_path=args.generator_model_path,
            device=args.device,
            state_dim=agent_state_dim,
            action_dim=args.agent_action_dim,
        )
    except FileNotFoundError as e:
        print(f"Error loading models: {e}. Please check that model paths are correct.")
        return
    except Exception as e:
        print(f"An unexpected error occurred while loading models: {e}")
        return

    print(f"\nGenerating motif sequence for prompt: '{args.prompt}'...")
    motif_sequence = generate_motif_sequence(
        prompt=args.prompt,
        agent=agent,
        text_encoder=text_encoder,
        max_steps=args.max_motifs,
        device=args.device
    )
    print(f"Generated motif sequence: {motif_sequence}")

    print("\nTranslating motif sequence to CadQuery script...")
    cadquery_script = generator.generate(prompt=args.prompt, motif_ids=motif_sequence)
    print("--- Generated CadQuery Script ---")
    print(cadquery_script)
    print("---------------------------------")

    print(f"\nExecuting script and saving model to {args.output_path}...")
    success, error_message = execute_cad_script(
        script_content=cadquery_script,
        output_stl_path=args.output_path
    )

    if success:
        print(f"\nSuccessfully generated and saved {args.output_path}")
    else:
        print(f"\nError during CadQuery execution: {error_message}")
        # Save the failed script for debugging purposes.
        script_path = args.output_path.with_suffix('.py')
        try:
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(cadquery_script)
            print(f"Failed CadQuery script saved to {script_path} for debugging.")
        except IOError as e:
            print(f"Could not save failed script: {e}")


if __name__ == "__main__":
    main()