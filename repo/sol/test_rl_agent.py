"""
test_rl_agent.py - Comprehensive unit tests for the DQN agent in the solitaire RL project.

This module contains extensive unit tests for the DQNAgent class, verifying
its interaction with game states, action selection, learning from experiences,
model persistence, and integration with the game logic. Tests cover all major
aspects including epsilon-greedy exploration, target network synchronization,
replay memory functionality, and performance under various conditions.
"""

import os
import sys
import pytest
import torch
import numpy as np
import tempfile
import random
from collections import deque
from unittest.mock import patch, MagicMock

from game.solitaire import SolitaireGame, Card
from rl.agent import DQNAgent
from utils.state_encoder import get_state_representation
from utils.action_utils import get_valid_actions, apply_action
import config

class TestDQNAgentInitialization:
    """Test suite for DQNAgent initialization and basic properties."""

    def test_agent_creation(self):
        """Test that the agent is created correctly with all required components."""
        state_size = 100
        action_size = 50
        device = 'cpu'
        agent = DQNAgent(state_size, action_size, device)

        assert agent.state_size == state_size
        assert agent.action_size == action_size
        assert agent.device == device
        assert agent.gamma == config.GAMMA
        assert agent.epsilon == config.EPSILON_START
        assert agent.steps == 0

    def test_model_architecture(self):
        """Test that the neural network models are constructed correctly."""
        state_size = 200
        action_size = 100
        agent = DQNAgent(state_size, action_size, 'cpu')

        assert agent.model is not None
        assert isinstance(agent.model, torch.nn.Module)
        assert agent.target_model is not None
        assert isinstance(agent.target_model, torch.nn.Module)

    def test_optimizer_initialization(self):
        """Test that the optimizer is initialized correctly."""
        agent = DQNAgent(100, 50, 'cpu')
        assert agent.optimizer is not None
        assert isinstance(agent.optimizer, torch.optim.Adam)

    def test_memory_initialization(self):
        """Test that the replay memory is initialized correctly."""
        agent = DQNAgent(100, 50, 'cpu')
        assert agent.memory is not None
        assert len(agent.memory) == 0

class TestEpsilonGreedyActionSelection:
    """Test suite for epsilon-greedy action selection mechanism."""

    def test_purely_random_action_selection(self):
        """Test action selection when epsilon is 1.0 (always random)."""
        agent = DQNAgent(100, 50, 'cpu')
        agent.epsilon = 1.0

        state = np.random.random(100)
        valid_actions = [0, 1, 2, 3, 4]

        with patch('random.choice') as mock_choice:
            mock_choice.return_value = 2
            action_idx, epsilon_used = agent.get_action(state, valid_actions)

            assert action_idx == 2
            assert epsilon_used == 1.0
            mock_choice.assert_called_once_with(valid_actions)

    def test_purely_greedy_action_selection(self):
        """Test action selection when epsilon is 0.0 (always greedy)."""
        agent = DQNAgent(100, 50, 'cpu')
        agent.epsilon = 0.0

        state = np.random.random(100)
        valid_actions = [0, 1, 2, 3, 4]

        with patch.object(agent.model, 'forward') as mock_forward:
            mock_forward.return_value = torch.tensor([[0.1, 0.5, 0.9, 0.2, 0.3]])

            action_idx, epsilon_used = agent.get_action(state, valid_actions)

            assert action_idx == 2
            assert epsilon_used == 0.0

    def test_mixed_exploration_exploitation(self):
        """Test action selection with mixed exploration/exploitation."""
        agent = DQNAgent(100, 50, 'cpu')
        agent.epsilon = 0.3

        state = np.random.random(100)
        valid_actions = [0, 1, 2, 3, 4]

        results = []
        for _ in range(100):
            with patch('random.random') as mock_random:
                mock_random.return_value = 0.2
                with patch('random.choice') as mock_choice:
                    mock_choice.return_value = 3
                    action_idx, epsilon_used = agent.get_action(state, valid_actions)
                    results.append((action_idx, epsilon_used))

        assert any(r[0] == 3 for r in results)

class TestExperienceReplay:
    """Test suite for experience replay functionality."""

    def test_store_experience(self):
        """Test storing a single experience in replay memory."""
        agent = DQNAgent(100, 50, 'cpu')

        state = np.random.random(100)
        action = 5
        reward = 10.0
        next_state = np.random.random(100)
        done = False

        initial_memory_size = len(agent.memory)
        agent.remember(state, action, reward, next_state, done)

        assert len(agent.memory) == initial_memory_size + 1

        stored_experience = agent.memory.memory[-1]
        np.testing.assert_array_equal(stored_experience[0], state)
        assert stored_experience[1] == action
        assert stored_experience[2] == reward
        np.testing.assert_array_equal(stored_experience[3], next_state)
        assert stored_experience[4] == done

    def test_replay_training_updates_weights(self):
        """Test that replay training actually updates model weights."""
        agent = DQNAgent(50, 20, 'cpu')

        for _ in range(64):
            state = np.random.random(50)
            action = np.random.randint(0, 20)
            reward = np.random.random()
            next_state = np.random.random(50)
            done = np.random.random() < 0.5
            agent.remember(state, action, reward, next_state, done)

        initial_weights = {}
        for name, param in agent.model.named_parameters():
            initial_weights[name] = param.data.clone()

        loss = agent.replay(64)

        assert loss is not None
        assert isinstance(loss, float)
        assert loss >= 0

        for name, param in agent.model.named_parameters():
            assert not torch.equal(param.data, initial_weights[name])

    def test_insufficient_memory_for_replay(self):
        """Test handling when replay memory has insufficient experiences."""
        agent = DQNAgent(100, 50, 'cpu')

        for _ in range(10):
            state = np.random.random(100)
            action = np.random.randint(0, 50)
            reward = np.random.random()
            next_state = np.random.random(100)
            done = False
            agent.remember(state, action, reward, next_state, done)

        loss = agent.replay(64)
        assert loss is None

    def test_large_batch_replay_performance(self):
        """Test replay performance with large batch sizes."""
        agent = DQNAgent(200, 100, 'cpu')

        batch_size = 128
        for _ in range(batch_size * 2):
            state = np.random.random(200)
            action = np.random.randint(0, 100)
            reward = np.random.random()
            next_state = np.random.random(200)
            done = np.random.random() < 0.1
            agent.remember(state, action, reward, next_state, done)

        import time
        start_time = time.time()
        loss = agent.replay(batch_size)
        end_time = time.time()

        assert loss is not None
        assert end_time - start_time < 2.0

class TestTargetNetworkSynchronization:
    """Test suite for target network update mechanism."""

    def test_initial_target_network_sync(self):
        """Test that target network is initially synchronized with main network."""
        agent = DQNAgent(100, 50, 'cpu')

        for main_param, target_param in zip(agent.model.parameters(), agent.target_model.parameters()):
            assert torch.equal(main_param.data, target_param.data)

    def test_manual_target_network_update(self):
        """Test manual target network update."""
        agent = DQNAgent(100, 50, 'cpu')

        with torch.no_grad():
            for param in agent.model.parameters():
                param.fill_(1.0)

        with torch.no_grad():
            for param in agent.target_model.parameters():
                param.fill_(0.0)

        agent._update_target_network()

        for main_param, target_param in zip(agent.model.parameters(), agent.target_model.parameters()):
            assert torch.equal(main_param.data, target_param.data)

    def test_automatic_target_network_update(self):
        """Test automatic target network update during training."""
        agent = DQNAgent(50, 20, 'cpu')

        for _ in range(64):
            state = np.random.random(50)
            action = np.random.randint(0, 20)
            reward = np.random.random()
            next_state = np.random.random(50)
            done = np.random.random() < 0.5
            agent.remember(state, action, reward, next_state, done)

        with torch.no_grad():
            for param in agent.target_model.parameters():
                param.fill_(999.0)

        agent.steps = config.TARGET_NETWORK_UPDATE_FREQ - 1
        loss = agent.replay(64)

        for main_param, target_param in zip(agent.model.parameters(), agent.target_model.parameters()):
            assert torch.allclose(main_param.data, target_param.data, atol=1e-6)

    def test_target_network_update_frequency(self):
        """Test that target network updates occur at the correct frequency."""
        agent = DQNAgent(50, 20, 'cpu')

        for _ in range(64):
            state = np.random.random(50)
            action = np.random.randint(0, 20)
            reward = np.random.random()
            next_state = np.random.random(50)
            done = np.random.random() < 0.5
            agent.remember(state, action, reward, next_state, done)

        agent.steps = config.TARGET_NETWORK_UPDATE_FREQ - 1

        with torch.no_grad():
            for param in agent.target_model.parameters():
                param.fill_(888.0)

        agent.replay(64)
        assert agent.steps == config.TARGET_NETWORK_UPDATE_FREQ

        agent.replay(64)
        assert agent.steps == config.TARGET_NETWORK_UPDATE_FREQ + 1

        for main_param, target_param in zip(agent.model.parameters(), agent.target_model.parameters()):
            assert torch.allclose(main_param.data, target_param.data, atol=1e-6)

class TestEpsilonDecay:
    """Test suite for epsilon decay mechanism."""

    def test_epsilon_decay_from_start(self):
        """Test epsilon decay from starting value."""
        agent = DQNAgent(100, 50, 'cpu')
        initial_epsilon = agent.epsilon

        agent.decay_epsilon()

        expected_epsilon = max(config.EPSILON_END, initial_epsilon * config.EPSILON_DECAY)
        assert agent.epsilon == expected_epsilon

    def test_epsilon_never_below_minimum(self):
        """Test that epsilon doesn't go below the minimum value."""
        agent = DQNAgent(100, 50, 'cpu')
        agent.epsilon = config.EPSILON_END

        for _ in range(10):
            agent.decay_epsilon()
            assert agent.epsilon == config.EPSILON_END

    def test_epsilon_reset_functionality(self):
        """Test that epsilon can be reset to starting value."""
        agent = DQNAgent(100, 50, 'cpu')

        agent.decay_epsilon()
        decayed_epsilon = agent.epsilon

        agent.reset_epsilon()

        assert agent.epsilon == config.EPSILON_START
        assert agent.epsilon > decayed_epsilon

class TestModelPersistence:
    """Test suite for model saving and loading."""

    def test_save_and_load_model(self):
        """Test complete model saving and loading cycle."""
        agent = DQNAgent(100, 50, 'cpu')

        for _ in range(32):
            state = np.random.random(100)
            action = np.random.randint(0, 50)
            reward = np.random.random()
            next_state = np.random.random(100)
            done = np.random.random() < 0.5
            agent.remember(state, action, reward, next_state, done)
        agent.replay(32)

        original_epsilon = agent.epsilon
        original_steps = agent.steps

        with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp_file:
            model_path = tmp_file.name

        try:
            agent.save_model(model_path)
            assert os.path.exists(model_path)

            new_agent = DQNAgent(100, 50, 'cpu')
            new_agent.load_model(model_path)

            for orig_param, loaded_param in zip(agent.model.parameters(), new_agent.model.parameters()):
                assert torch.equal(orig_param.data, loaded_param.data)

            for orig_param, loaded_param in zip(agent.target_model.parameters(), new_agent.target_model.parameters()):
                assert torch.equal(orig_param.data, loaded_param.data)

            assert new_agent.epsilon == original_epsilon
            assert new_agent.steps == original_steps

        finally:
            if os.path.exists(model_path):
                os.unlink(model_path)

    def test_load_nonexistent_model(self):
        """Test handling when trying to load a non-existent model."""
        agent = DQNAgent(100, 50, 'cpu')

        with pytest.raises(FileNotFoundError):
            agent.load_model('nonexistent_model.pth')

class TestQValueComputations:
    """Test suite for Q-value computation methods."""

    def test_get_q_values_output_format(self):
        """Test that get_q_values returns correct output format."""
        agent = DQNAgent(100, 50, 'cpu')

        state = np.random.random(100)
        q_values = agent.get_q_values(state)

        assert isinstance(q_values, np.ndarray)
        assert q_values.shape == (50,)
        assert q_values.dtype == np.float32 or q_values.dtype == np.float64

    def test_q_value_consistency(self):
        """Test that Q-values are consistent across calls."""
        agent = DQNAgent(100, 50, 'cpu')

        state = np.random.random(100)

        q_values1 = agent.get_q_values(state)
        q_values2 = agent.get_q_values(state)

        np.testing.assert_array_almost_equal(q_values1, q_values2)

    def test_q_values_used_in_action_selection(self):
        """Test that Q-values are properly used in greedy action selection."""
        agent = DQNAgent(100, 50, 'cpu')
        agent.epsilon = 0.0

        state = np.random.random(100)
        valid_actions = [10, 20, 30, 40]

        q_values = agent.get_q_values(state)

        action_idx, _ = agent.get_action(state, valid_actions)

        valid_q_values = [q_values[action] for action in valid_actions]
        expected_action = valid_actions[np.argmax(valid_q_values)]

        assert action_idx == expected_action

class TestDQNAgentStatistics:
    """Test suite for agent statistics and monitoring."""

    def test_get_stats_output_format(self):
        """Test that get_stats returns a properly formatted dictionary."""
        agent = DQNAgent(100, 50, 'cpu')
        agent.epsilon = 0.75
        agent.steps = 500

        stats = agent.get_stats()

        assert isinstance(stats, dict)
        assert 'epsilon' in stats
        assert 'steps' in stats
        assert 'memory_size' in stats

        assert stats['epsilon'] == 0.75
        assert stats['steps'] == 500
        assert stats['memory_size'] == 0

    def test_stats_update_correctly(self):
        """Test that statistics update correctly as agent operates."""
        agent = DQNAgent(100, 50, 'cpu')

        initial_stats = agent.get_stats()
        assert initial_stats['epsilon'] == config.EPSILON_START
        assert initial_stats['steps'] == 0
        assert initial_stats['memory_size'] == 0

        for _ in range(10):
            state = np.random.random(100)
            action = np.random.randint(0, 50)
            reward = np.random.random()
            next_state = np.random.random(100)
            done = False
            agent.remember(state, action, reward, next_state, done)

        agent.decay_epsilon()
        agent.steps = 100

        updated_stats = agent.get_stats()
        assert updated_stats['epsilon'] < config.EPSILON_START
        assert updated_stats['steps'] == 100
        assert updated_stats['memory_size'] == 10

class TestDQNAgentGameIntegration:
    """Integration tests for DQN agent with solitaire game."""

    def test_agent_with_real_game_state(self):
        """Test agent interaction with actual solitaire game."""
        game = SolitaireGame()
        state = get_state_representation(game)

        agent = DQNAgent(len(state), 50, 'cpu')

        valid_actions = get_valid_actions(game)
        if valid_actions:
            action_indices = list(range(len(valid_actions)))
            action_idx, epsilon = agent.get_action(state, action_indices)