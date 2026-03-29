"""
Model Persistence and Training Logging Utilities for Blackjack Reinforcement Learning Agent.

This module provides utility functions for saving and loading the reinforcement learning
agent's state, as well as logging training statistics to CSV files. It includes functions
for serializing and deserializing the agent's Q-table and parameters using Python's pickle
module, managing file operations, and handling training statistics logging.

The persistence functionality supports:
- Saving the complete agent state (Q-table, hyperparameters, and configuration)
- Loading agent state from saved files
- Error handling for file operations with detailed error messages
- Directory creation for model storage
- Model validation and metadata extraction

The logging functionality includes:
- Appending training statistics to CSV files
- Automatic file and directory creation
- Error handling for file operations
"""

import pickle
import csv
import os
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime
import logging
import config

# Configure logging
logger = logging.getLogger(__name__)

def save_model(agent: Any, filepath: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
    """
    Save the RL agent's state to a file using pickle serialization.

    This function serializes the agent's Q-table, configuration parameters, and optional
    metadata to a pickle file. It automatically creates directories if they don't exist
    and includes version information for compatibility checking.

    Args:
        agent: The RL agent instance to save (must have q_table and other required attributes)
        filepath: Path where the model should be saved (can be absolute or relative)
        metadata: Optional additional metadata to include with the model (training stats, etc.)

    Returns:
        bool: True if save was successful, False otherwise

    Raises:
        ValueError: If agent doesn't have required attributes
        PermissionError: If file cannot be written due to permission issues
        IOError: If file cannot be written due to other I/O issues
    """
    try:
        # Validate agent has required attributes
        required_attrs = ['q_table', 'learning_rate', 'discount_factor', 'exploration_rate',
                         'min_exploration_rate', 'exploration_decay']
        missing_attrs = [attr for attr in required_attrs if not hasattr(agent, attr)]

        if missing_attrs:
            raise ValueError(f"Agent missing required attributes: {', '.join(missing_attrs)}")

        # Create directory if it doesn't exist
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            try:
                os.makedirs(directory, exist_ok=True)
            except Exception as e:
                raise IOError(f"Failed to create directory {directory}: {e}")

        # Prepare comprehensive model data with version information
        model_data = {
            'version': '1.0',
            'timestamp': datetime.now().isoformat(),
            'agent_type': 'BlackjackRLAgent',
            'project_name': 'Blackjack Reinforcement Learning Agent',
            'q_table': agent.q_table,
            'config': {
                'learning_rate': agent.learning_rate,
                'discount_factor': agent.discount_factor,
                'exploration_rate': agent.exploration_rate,
                'min_exploration_rate': agent.min_exploration_rate,
                'exploration_decay': agent.exploration_decay,
                'state_discretization': getattr(agent, 'state_discretization', config.RL_CONFIG.get("state_discretization", {})),
                'reward_scaling': getattr(agent, 'reward_scaling', config.RL_CONFIG.get("reward_scaling", 0.1)),
                'action_space': getattr(agent, 'action_space', list(range(1, 13))),
            },
            'discretization_bins': {
                'count_bins': getattr(agent, 'count_bins', []),
                'bankroll_bins': getattr(agent, 'bankroll_bins', []),
                'bet_size_bins': getattr(agent, 'bet_size_bins', []),
            },
            'metadata': metadata or {}
        }

        # Add optional training statistics
        if hasattr(agent, 'total_training_reward'):
            model_data['metadata']['total_training_reward'] = agent.total_training_reward

        # Write to file with highest protocol for better performance
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            raise IOError(f"Failed to write to file {filepath}: {e}")

        logger.info(f"Model successfully saved to {filepath}")
        if 'total_training_reward' in model_data['metadata']:
            logger.info(f"Training reward: {model_data['metadata']['total_training_reward']:.2f}")
        return True

    except (ValueError, PermissionError, IOError):
        raise  # Re-raise specific exceptions
    except Exception as e:
        logger.error(f"Unexpected error while saving model: {e}")
        raise IOError(f"Unexpected error while saving model: {e}")

def load_model(filepath: str, agent_class: Optional[Any] = None) -> Tuple[Optional[Any], Dict[str, Any]]:
    """
    Load an RL agent's state from a pickle file.

    This function deserializes a previously saved agent state, including the Q-table
    and configuration parameters. It performs version checking and data validation
    to ensure compatibility and integrity.

    Args:
        filepath: Path to the model file to load (can be absolute or relative)
        agent_class: Optional agent class to instantiate (if not provided, returns raw data dict)

    Returns:
        Tuple containing:
        - Loaded agent instance (or None if agent_class not provided or instantiation fails)
        - Dictionary with metadata and configuration information

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file is corrupted, incompatible, or invalid format
        PermissionError: If access to the file is denied
    """
    try:
        # Check if file exists with detailed error message
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}. "
                                   f"Please ensure the file exists or train a new model.")

        # Check file readability
        if not os.access(filepath, os.R_OK):
            raise PermissionError(f"Access denied: Cannot read file {filepath}")

        # Read from file with error handling
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
        except pickle.UnpicklingError as e:
            raise ValueError(f"Unable to unpickle model file {filepath}. File may be corrupted. Error: {e}")
        except Exception as e:
            raise IOError(f"Failed to read model file {filepath}: {e}")

        # Validate file format and structure
        if not isinstance(model_data, dict):
            raise ValueError(f"Invalid model file format: Expected dictionary, got {type(model_data)}")

        required_keys = ['version', 'agent_type', 'q_table', 'config']
        missing_keys = [key for key in required_keys if key not in model_data]

        if missing_keys:
            raise ValueError(f"Invalid model file format: Missing required keys {', '.join(missing_keys)}")

        # Version compatibility check
        if model_data['version'] != '1.0':
            logger.warning(f"Model version {model_data['version']} "
                          f"may not be fully compatible with current version 1.0")

        # Extract data
        q_table = model_data.get('q_table', {})
        config_data = model_data.get('config', {})
        discretization_bins = model_data.get('discretization_bins', {})
        metadata = model_data.get('metadata', {})

        # Create agent instance if class provided
        agent = None
        if agent_class:
            try:
                # Create agent with loaded configuration
                agent = agent_class(
                    learning_rate=config_data.get('learning_rate', 0.1),
                    discount_factor=config_data.get('discount_factor', 0.95),
                    initial_exploration_rate=config_data.get('exploration_rate', 0.5),
                    min_exploration_rate=config_data.get('min_exploration_rate', 0.01),
                    exploration_decay=config_data.get('exploration_decay', 0.9995),
                    state_discretization=config_data.get('state_discretization', config.RL_CONFIG.get("state_discretization", {})),
                    reward_scaling=config_data.get('reward_scaling', config.RL_CONFIG.get("reward_scaling", 0.1))
                )

                # Restore agent state
                agent.q_table = q_table
                agent.action_space = config_data.get('action_space', list(range(1, 13)))

                # Restore discretization bins
                if 'count_bins' in discretization_bins:
                    agent.count_bins = discretization_bins['count_bins']
                if 'bankroll_bins' in discretization_bins:
                    agent.bankroll_bins = discretization_bins['bankroll_bins']
                if 'bet_size_bins' in discretization_bins:
                    agent.bet_size_bins = discretization_bins['bet_size_bins']

            except Exception as e:
                logger.error(f"Failed to create agent instance: {e}")
                agent = None

        # Prepare comprehensive return metadata
        return_metadata = {
            'version': model_data['version'],
            'timestamp': model_data.get('timestamp', 'Unknown'),
            'agent_type': model_data.get('agent_type', 'Unknown'),
            'project_name': model_data.get('project_name', 'Unknown Project'),
            'q_table_size': len(q_table),
            'total_states': len(q_table),
            'total_q_entries': sum(len(actions) for actions in q_table.values()),
            'config': config_data,
            'metadata': metadata,
            'file_size_bytes': os.path.getsize(filepath),
            'last_modified': datetime.fromtimestamp(os.path.getmtime(filepath)).isoformat()
        }

        # Add training statistics if available
        if 'total_training_reward' in metadata:
            return_metadata['total_training_reward'] = metadata['total_training_reward']

        logger.info(f"Model successfully loaded from {filepath}")
        logger.info(f"Q-table size: {return_metadata['q_table_size']} states")
        if 'total_training_reward' in return_metadata:
            logger.info(f"Previous training reward: {return_metadata['total_training_reward']:.2f}")

        return agent, return_metadata

    except (FileNotFoundError, ValueError, PermissionError):
        raise  # Re-raise specific exceptions
    except Exception as e:
        logger.error(f"Failed to load model from {filepath}: {e}")
        raise ValueError(f"Failed to load model: {e}")

def log_training_stats(filepath: str, episode: int, stats: Dict[str, Any]) -> bool:
    """
    Append training statistics to a CSV file for analysis and visualization.

    This function logs training statistics to a CSV file, automatically creating
    the file with headers if it doesn't exist. It handles various data types and
    provides robust error handling for file operations.

    Args:
        filepath: Path to the CSV file for logging (e.g., 'logs/training_stats.csv')
        episode: Current episode number
        stats: Dictionary containing training statistics (bankroll, win_rate, profit, etc.)

    Returns:
        bool: True if logging was successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            try:
                os.makedirs(directory, exist_ok=True)
            except Exception as e:
                raise IOError(f"Failed to create directory {directory}: {e}")

        # Prepare data for CSV writing
        row_data = {
            'timestamp': datetime.now().isoformat(),
            'episode': episode
        }
        row_data.update(stats)

        # Check if file exists to determine if headers need to be written
        file_exists = os.path.exists(filepath)

        try:
            with open(filepath, 'a', newline='') as f:
                fieldnames = list(row_data.keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)

                # Write header if file doesn't exist
                if not file_exists or f.tell() == 0:
                    writer.writeheader()

                # Write the data row
                writer.writerow(row_data)

        except PermissionError:
            raise PermissionError(f"Access denied: Cannot write to {filepath}")
        except Exception as e:
            raise IOError(f"Failed to write to CSV file {filepath}: {e}")

        logger.debug(f"Training stats logged to {filepath}")
        return True

    except (PermissionError, IOError):
        raise
    except Exception as e:
        logger.error(f"Unexpected error while logging training stats: {e}")
        raise IOError(f"Unexpected error while logging training stats: {e}")

def get_model_info(filepath: str) -> Dict[str, Any]:
    """
    Get information about a saved model without loading the full agent.

    This function reads only the metadata and configuration from a model file,
    providing quick access to model information without the overhead of loading
    the entire Q-table.

    Args:
        filepath: Path to the model file

    Returns:
        Dictionary containing model metadata, configuration, and file information

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file is corrupted or invalid
        PermissionError: If access to the file is denied
    """
    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        if not os.access(filepath, os.R_OK):
            raise PermissionError(f"Access denied: Cannot read file {filepath}")

        # Load only metadata without full processing
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
        except Exception as e:
            raise ValueError(f"Unable to load model file {filepath}: {e}")

        # Extract relevant information
        q_table = model_data.get('q_table', {})
        config_data = model_data.get('config', {})
        metadata = model_data.get('metadata', {})
        discretization_bins = model_data.get('discretization_bins', {})

        info = {
            'version': model_data.get('version', 'Unknown'),
            'timestamp': model_data.get('timestamp', 'Unknown'),
            'agent_type': model_data.get('agent_type', 'Unknown'),
            'project_name': model_data.get('project_name', 'Unknown Project'),
            'q_table_size': len(q_table),
            'total_q_entries': sum(len(actions) for actions in q_table.values()),
            'learning_rate': config_data.get('learning_rate', 'N/A'),
            'discount_factor': config_data.get('discount_factor', 'N/A'),
            'exploration_rate': config_data.get('exploration_rate', 'N/A'),
            'action_space_size': len(config_data.get('action_space', [])),
            'metadata': metadata,
            'file_size_bytes': os.path.getsize(filepath),
            'last_modified': datetime.fromtimestamp(os.path.getmtime(filepath)).isoformat(),
            'discretization_bins': discretization_bins
        }

        return info

    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise ValueError(f"Failed to get model info: {e}")

def validate_model_data(model_data: Dict) -> bool:
    """
    Validate that the loaded model data contains all required components and valid values.

    This function performs comprehensive validation of loaded model data to ensure
    it's properly formatted and contains valid values for all required fields.

    Args:
        model_data: Dictionary containing model data to validate

    Returns:
        True if the model data is valid and properly formatted, False otherwise
    """
    try:
        # Check for required top-level keys
        required_keys = ['version', 'q_table', 'config']
        for key in required_keys:
            if key not in model_data:
                return False

        config_data = model_data['config']

        # Check for required config keys
        config_required_keys = [
            'learning_rate', 'discount_factor', 'exploration_rate',
            'min_exploration_rate', 'exploration_decay',
            'state_discretization', 'reward_scaling', 'action_space'
        ]

        for key in config_required_keys:
            if key not in config_data:
                return False

        # Validate data types and value ranges
        # Q-table validation
        if not isinstance(model_data['q_table'], dict):
            return False

        # Numeric values validation
        learning_rate = config_data['learning_rate']
        discount_factor = config_data['discount_factor']
        exploration_rate = config_data['exploration_rate']
        min_exploration_rate = config_data['min_exploration_rate']
        exploration_decay = config_data['exploration_decay']

        if not all(isinstance(val, (int, float)) for val in [
            learning_rate, discount_factor, exploration_rate,
            min_exploration_rate, exploration_decay
        ]):
            return False

        # Value range validation
        if not all(0 <= val <= 1 for val in [
            learning_rate, discount_factor, exploration_rate,
            min_exploration_rate, exploration_decay
        ]):
            return False

        # Action space validation
        action_space = config_data['action_space']
        if not isinstance(action_space, list) or len(action_space) == 0:
            return False

        # All validations passed
        return True

    except Exception:
        # If any validation step fails, return False
        return False

def get_logged_stats(filepath: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Read logged training statistics from a CSV file.

    Args:
        filepath: Path to the CSV file containing training statistics
        limit: Optional limit on number of rows to return (latest entries)

    Returns:
        List of dictionaries containing training statistics

    Raises:
        FileNotFoundError: If the log file doesn't exist
        ValueError: If the