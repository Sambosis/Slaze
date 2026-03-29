"""
test_rl_agent.py - Unit tests for the DQN agent in the solitaire RL project.

This module contains comprehensive unit tests for the DQNAgent class, verifying
its interaction with game states, action selection, learning from experiences, and
model persistence. Tests cover epsilon-greedy exploration, target network
synchronization, model saving/loading, and integration with the game logic.
"""

import pytest
import torch
import numpy as np
import os
import tempfile
import random
from unittest.mock import patch

from rl.agent import DQNAgent
from game.solitaire import SolitaireGame, Card
from utils.state_encoder import get_state_representation
from utils.action_utils import get_valid_actions, apply_action
import config

class TestDQNAgent:
    """Test suite for the DQNAgent class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.state_size = 100
        self.action_size = 50
        self.device = 'cpu'
        self.agent = DQNAgent(self.state_size, self.action_size, self.device)

    def test_initialization(self):
        """Test that the agent is initialized correctly."""
        assert self.agent.state_size == self.state_size
        assert self.agent.action_size == self.action_size
        assert self.agent.device == self.device
        assert self.agent.epsilon == config.EPSILON_START
        assert self.agent.steps == 0
        assert self.agent.model is not None
        assert self.agent.target_model is not None
        assert isinstance(self.agent.model, torch.nn.Module)
        assert self.agent.optimizer is not None
        assert self.agent.memory is not None
        assert len(self.agent.memory) == 0

    def test_epsilon_greedy_action_selection(self):
        """Test epsilon-greedy action selection."""
        # Test with epsilon = 1.0 (always random)
        self.agent.epsilon = 1.0
        state = np.random.random(self.state_size)
        valid_actions = [0, 1, 2, 3, 4]

        # Mock the model output to return predictable Q-values
        with patch.object(self.agent.model, 'forward') as mock_forward:
            mock_forward.return_value = torch.tensor([[0.1, 0.5, 0.9, 0.2, 0.3]])

            action_idx, epsilon_used = self.agent.get_action(state, valid_actions)
            assert action_idx in valid_actions
            assert epsilon_used == 1.0

        # Test with epsilon = 0.0 (always greedy)
        self.agent.epsilon = 0.0

        with patch.object(self.agent.model, 'forward') as mock_forward:
            mock_forward.return_value = torch.tensor([[0.1, 0.5, 0.9, 0.2, 0.3]])

            action_idx, epsilon_used = self.agent.get_action(state, valid_actions)
            assert action_idx == 2  # Action 2 has Q-value 0.9
            assert epsilon_used == 0.0

    def test_remember_experience(self):
        """Test storing experiences in replay memory."""
        state = np.random.random(self.state_size)
        action = 3
        reward = 10.0
        next_state = np.random.random(self.state_size)
        done = False

        initial_size = len(self.agent.memory)
        self.agent.remember(state, action, reward, next_state, done)

        assert len(self.agent.memory) == initial_size + 1

        # Verify the stored experience
        stored_state, stored_action, stored_reward, stored_next_state, stored_done = self.agent.memory.memory[-1]
        np.testing.assert_array_equal(stored_state, state)
        assert stored_action == action
        assert stored_reward == reward
        np.testing.assert_array_equal(stored_next_state, next_state)
        assert stored_done == done

    def test_replay_training(self):
        """Test the replay training method."""
        # Add enough experiences to the memory
        batch_size = 32
        for _ in range(batch_size + 5):
            state = np.random.random(self.state_size)
            action = np.random.randint(0, self.action_size)
            reward = np.random.random()
            next_state = np.random.random(self.state_size)
            done = np.random.random() < 0.5
            self.agent.remember(state, action, reward, next_state, done)

        # Save initial model weights
        initial_weights = {}
        for name, param in self.agent.model.named_parameters():
            initial_weights[name] = param.data.clone()

        # Perform replay training
        loss = self.agent.replay(batch_size)

        # Check that loss is returned
        assert loss is not None
        assert isinstance(loss, float)

        # Check that model weights have changed (training occurred)
        for name, param in self.agent.model.named_parameters():
            assert not torch.equal(param.data, initial_weights[name])

    def test_replay_insufficient_memory(self):
        """Test replay when there are insufficient experiences in memory."""
        # Try to replay with insufficient memory
        loss = self.agent.replay(32)
        assert loss is None

    def test_epsilon_decay(self):
        """Test epsilon decay."""
        initial_epsilon = self.agent.epsilon
        self.agent.decay_epsilon()

        expected_epsilon = max(config.EPSILON_END, initial_epsilon * config.EPSILON_DECAY)
        assert self.agent.epsilon == expected_epsilon

        # Test that epsilon doesn't go below epsilon_end
        self.agent.epsilon = config.EPSILON_END
        self.agent.decay_epsilon()
        assert self.agent.epsilon == config.EPSILON_END

    def test_target_network_update(self):
        """Test target network synchronization."""
        # Make models different
        with torch.no_grad():
            for param in self.agent.model.parameters():
                param.fill_(1.0)

        for param in self.agent.target_model.parameters():
            param.fill_(0.0)

        # Update target network
        self.agent._update_target_network()

        # Check that target network now matches main network
        for main_param, target_param in zip(self.agent.model.parameters(), self.agent.target_model.parameters()):
            assert torch.equal(main_param.data, target_param.data)

    def test_save_and_load_model(self):
        """Test saving and loading the model."""
        # Create a temporary file for saving
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp_file:
            model_path = tmp_file.name

        try:
            # Train the model a bit to change its weights
            state = np.random.random(self.state_size)
            action = 0
            reward = 1.0
            next_state = np.random.random(self.state_size)
            done = False
            self.agent.remember(state, action, reward, next_state, done)
            self.agent.replay(1)

            # Save the model
            initial_epsilon = self.agent.epsilon
            initial_steps = self.agent.steps
            self.agent.save_model(model_path)

            # Create a new agent and load the model
            new_agent = DQNAgent(self.state_size, self.action_size, self.device)
            new_agent.load_model(model_path)

            # Check that the models have the same weights
            for main_param, target_param in zip(self.agent.model.parameters(), new_agent.model.parameters()):
                assert torch.equal(main_param.data, target_param.data)

            # Check that other attributes are loaded correctly
            assert new_agent.epsilon == initial_epsilon
            assert new_agent.steps == initial_steps

        finally:
            # Clean up the temporary file
            if os.path.exists(model_path):
                os.unlink(model_path)

    def test_get_q_values(self):
        """Test getting Q-values for a state."""
        state = np.random.random(self.state_size)
        q_values = self.agent.get_q_values(state)

        assert isinstance(q_values, np.ndarray)
        assert q_values.shape == (self.action_size,)

    def test_reset_epsilon(self):
        """Test resetting epsilon to its starting value."""
        self.agent.epsilon = 0.1
        self.agent.reset_epsilon()
        assert self.agent.epsilon == config.EPSILON_START

    def test_get_stats(self):
        """Test getting agent statistics."""
        self.agent.epsilon = 0.5
        self.agent.steps = 100

        stats = self.agent.get_stats()

        assert isinstance(stats, dict)
        assert stats['epsilon'] == 0.5
        assert stats['steps'] == 100
        assert 'memory_size' in stats

class TestDQNIntegration:
    """Integration tests for the DQN agent with the solitaire game."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.game = SolitaireGame()
        state_size = 100
        action_size = 1000
        self.agent = DQNAgent(state_size, action_size, 'cpu')

    def test_agent_game_interaction(self):
        """Test that the agent can interact with the game."""
        # Get initial state
        state = get_state_representation(self.game)
        assert isinstance(state, np.ndarray)

        # Get valid actions
        valid_actions = get_valid_actions(self.game)
        assert isinstance(valid_actions, list)

        # Get action from agent
        if valid_actions:
            action_indices = list(range(len(valid_actions)))
            action_idx, epsilon = self.agent.get_action(state, action_indices)

            if action_idx < len(valid_actions):
                action = valid_actions[action_idx]

                # Apply action to game
                reward = apply_action(self.game, action)
                assert isinstance(reward, (int, float))

                # Get next state
                next_state = get_state_representation(self.game)
                assert isinstance(next_state, np.ndarray)

                # Store experience
                self.agent.remember(state, action_idx, reward, next_state, False)

    def test_episode_simulation(self):
        """Test simulating a complete episode."""
        max_steps = 20
        steps = 0
        total_reward = 0

        while steps < max_steps:
            # Get state and valid actions
            state = get_state_representation(self.game)
            valid_actions = get_valid_actions(self.game)

            if not valid_actions:
                break

            # Get action from agent
            action_indices = list(range(len(valid_actions)))
            action_idx, _ = self.agent.get_action(state, action_indices)

            if action_idx < len(valid_actions):
                action = valid_actions[action_idx]

                # Apply action
                reward = apply_action(self.game, action)
                total_reward += reward

                # Get next state
                next_state = get_state_representation(self.game)

                # Store experience
                done = self.game._check_win() or steps >= max_steps - 1
                self.agent.remember(state, action_idx, reward, next_state, done)

                # Train agent
                if len(self.agent.memory) >= 16:
                    self.agent.replay(16)

                # Check if episode is done
                if done:
                    break

            steps += 1

        assert steps <= max_steps
        assert isinstance(total_reward, (int, float))

class TestDQNPerformance:
    """Performance and stress tests for the DQN agent."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.state_size = 200
        self.action_size = 500
        self.agent = DQNAgent(self.state_size, self.action_size, 'cpu')

    def test_large_batch_replay(self):
        """Test replay with a large batch size."""
        # Fill memory with experiences
        batch_size = 128
        num_experiences = batch_size * 2
        for _ in range(num_experiences):
            state = np.random.random(self.state_size)
            action = np.random.randint(0, self.action_size)
            reward = np.random.random()
            next_state = np.random.random(self.state_size)
            done = np.random.random() < 0.1
            self.agent.remember(state, action, reward, next_state, done)

        # Time the replay operation
        import time
        start_time = time.time()
        loss = self.agent.replay(batch_size)
        end_time = time.time()

        assert loss is not None
        assert end_time - start_time < 1.0

    def test_memory_management(self):
        """Test memory management with large number of experiences."""
        # Add many experiences to test memory usage
        num_experiences = 1000
        for _ in range(num_experiences):
            state = np.random.random(self.state_size)
            action = np.random.randint(0, self.action_size)
            reward = np.random.random()
            next_state = np.random.random(self.state_size)
            done = False
            self.agent.remember(state, action, reward, next_state, done)

        # Check memory size
        assert len(self.agent.memory) == min(num_experiences, config.MEMORY_SIZE)

        # Check that old experiences are removed when memory is full
        initial_size = len(self.agent.memory)
        for _ in range(config.MEMORY_SIZE + 100):
            state = np.random.random(self.state_size)
            action = np.random.randint(0, self.action_size)
            reward = np.random.random()
            next_state = np.random.random(self.state_size)
            done = False
            self.agent.remember(state, action, reward, next_state, done)

        assert len(self.agent.memory) == config.MEMORY_SIZE
        assert len(self.agent.memory) == initial_size

if __name__ == "__main__":
    pytest.main([__file__, "-v"])