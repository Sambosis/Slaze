import torch
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
from typing import Dict, Tuple, Optional, Any
from collections import deque
from config import Config
from ai.models import DQN, ICMModel
from ai.utils import ReplayBuffer

class Agent:
    """
    Reinforcement Learning Agent implementing Double DQN + ICM.
    
    Handles:
    - Epsilon-greedy action selection
    - Experience storage in replay buffer
    - Double DQN training with target network
    - ICM intrinsic reward computation and loss integration
    
    Attributes:
        policy_net: Main DQN network for action selection
        target_net: Target DQN network for stable Q-learning targets
        icm: Intrinsic Curiosity Module for exploration rewards
        optimizer: Adam optimizer for policy network
        icm_optimizer: Separate optimizer for ICM
        replay_buffer: Experience replay memory
        epsilon: Current exploration rate
        steps: Total training steps counter
        device: Training device (cuda/cpu)
    """
    
    def __init__(self, config: Config, team_id: int):
        """
        Initialize the RL agent.
        
        Args:
            config: Global configuration object
            team_id: 0 for Red team, 1 for Blue team (for reward perspective)
        """
        self.config = config
        self.team_id = team_id
        self.device = config.DEVICE
        self.n_actions = config.N_ACTIONS
        
        # Networks
        self.policy_net = DQN(config).to(self.device)
        self.target_net = DQN(config).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.icm = ICMModel(config).to(self.device)
        
        # Optimizers
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.LR_DQN)
        self.icm_optimizer = optim.Adam(self.icm.parameters(), lr=config.LR_ICM)
        
        # Training state
        self.replay_buffer = ReplayBuffer(
            config.REPLAY_BUFFER_SIZE, 
            config.BATCH_SIZE
        )
        self.epsilon = config.EPSILON_START
        self.steps = 0
        self.target_update_counter = 0
        
    def get_action(self, state: np.ndarray, unit_ids: list[int]) -> Dict[int, int]:
        """
        Select actions for all units using epsilon-greedy policy.
        
        Args:
            state: Current game state tensor (channels, H, W)
            unit_ids: List of alive unit IDs for this agent
            
        Returns:
            actions: Dict mapping unit_id -> action_index (0-5)
        """
        if random.random() < self.epsilon:
            # Random actions for exploration
            actions = {uid: random.randint(0, self.n_actions - 1) for uid in unit_ids}
        else:
            # Greedy actions using policy network
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
                actions = {}
                for i, uid in enumerate(unit_ids):
                    action = q_values[0].argmax().item()
                    actions[uid] = action
        
        return actions
    
    def store_experience(
        self, 
        state: np.ndarray, 
        actions: Dict[int, int], 
        extrinsic_rewards: Dict[int, float], 
        next_state: np.ndarray, 
        done: bool
    ):
        """
        Store experience tuple in replay buffer with intrinsic rewards.
        
        Intrinsic rewards are computed per-unit based on ICM prediction error.
        Total reward = extrinsic + (eta * intrinsic)
        
        Args:
            state: Current state tensor
            actions: Dict of unit_id -> action taken
            extrinsic_rewards: Dict of unit_id -> extrinsic reward
            next_state: Next state tensor
            done: Episode termination flag
        """
        # Convert states to tensors for ICM
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        next_state_t = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        
        # Get representative action (first unit's action for simplicity)
        sample_action_id = list(actions.keys())[0] if actions else 0
        sample_action = actions[sample_action_id]
        action_t = torch.LongTensor([sample_action]).to(self.device)
        
        # Compute intrinsic reward using ICM
        with torch.no_grad():
            icm_output = self.icm(state_t, next_state_t, action_t)
            intrinsic_reward = self.icm.compute_intrinsic_reward(
                icm_output['state_feat'], 
                icm_output['next_state_feat'], 
                action_t
            ).mean().item()
        
        # Scale intrinsic reward
        intrinsic_reward *= self.config.ICM_ETA
        
        # Combine rewards for all units
        total_rewards = {}
        for uid, ext_reward in extrinsic_rewards.items():
            total_reward = ext_reward + intrinsic_reward
            total_rewards[uid] = total_reward
        
        # Store in buffer (using representative values)
        representative_action = list(actions.values())[0] if actions else 0
        representative_reward = np.mean(list(total_rewards.values()))
        
        self.replay_buffer.add(
            state=state,
            action=representative_action,
            reward=representative_reward,
            next_state=next_state,
            done=done
        )
    
    def learn(self) -> Dict[str, float]:
        """
        Perform one training step using Double DQN + ICM loss.
        
        Returns:
            losses: Dictionary containing DQN loss, ICM losses, and total loss
        """
        if len(self.replay_buffer) < self.config.BATCH_SIZE:
            return {}
        
        # Sample batch
        batch = self.replay_buffer.sample()
        state_batch = torch.FloatTensor(batch['states']).to(self.device)
        action_batch = torch.LongTensor(batch['actions']).to(self.device)
        reward_batch = torch.FloatTensor(batch['rewards']).to(self.device)
        next_state_batch = torch.FloatTensor(batch['next_states']).to(self.device)
        done_batch = torch.BoolTensor(batch['dones']).to(self.device)
        
        # === Double DQN Training ===
        self.policy_net.train()
        
        # Current Q-values (policy net)
        current_q_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        
        # Double DQN: Use policy net to select actions, target net to evaluate
        with torch.no_grad():
            next_actions = self.policy_net(next_state_batch).max(1)[1].unsqueeze(1)
            next_q_values = self.target_net(next_state_batch).gather(1, next_actions)
            target_q_values = reward_batch.unsqueeze(1) + (self.config.GAMMA * next_q_values * ~done_batch.unsqueeze(1))
        
        dqn_loss = F.mse_loss(current_q_values, target_q_values)
        
        # === ICM Training ===
        self.icm.train()
        icm_output = self.icm(state_batch, next_state_batch, action_batch)
        
        # Inverse model loss (predict action from state/next_state)
        inverse_loss = F.cross_entropy(icm_output['pred_action'], action_batch)
        
        # Forward model loss (predict next_state features)
        forward_loss = F.mse_loss(icm_output['pred_next_feat'], icm_output['next_state_feat'])
        
        # Combined ICM loss
        icm_loss = (1 - self.config.ICM_BETA) * inverse_loss + self.config.ICM_BETA * forward_loss
        
        # === Backpropagation ===
        # DQN loss
        self.optimizer.zero_grad()
        dqn_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # ICM loss
        self.icm_optimizer.zero_grad()
        icm_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.icm.parameters(), 1.0)
        self.icm_optimizer.step()
        
        # === Update counters ===
        self.steps += 1
        self.target_update_counter += 1
        
        # Epsilon decay
        self.epsilon = max(self.config.EPSILON_END, self.epsilon * self.config.EPSILON_DECAY)
        
        # Target network update
        if self.target_update_counter >= self.config.TARGET_UPDATE_FREQ:
            self._update_target_network()
            self.target_update_counter = 0
        
        return {
            'dqn_loss': dqn_loss.item(),
            'icm_loss': icm_loss.item(),
            'inverse_loss': inverse_loss.item(),
            'forward_loss': forward_loss.item(),
            'epsilon': self.epsilon
        }
    
    def _update_target_network(self):
        """Hard update of target network to policy network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save_models(self, path_prefix: str):
        """Save agent models to disk."""
        torch.save(self.policy_net.state_dict(), f"{path_prefix}_policy.pth")
        torch.save(self.target_net.state_dict(), f"{path_prefix}_target.pth")
        torch.save(self.icm.state_dict(), f"{path_prefix}_icm.pth")
        torch.save({
            'epsilon': self.epsilon,
            'steps': self.steps
        }, f"{path_prefix}_agent_state.pth")
    
    def load_models(self, path_prefix: str):
        """Load agent models from disk."""
        self.policy_net.load_state_dict(torch.load(f"{path_prefix}_policy.pth", map_location=self.device))
        self.target_net.load_state_dict(torch.load(f"{path_prefix}_target.pth", map_location=self.device))
        self.icm.load_state_dict(torch.load(f"{path_prefix}_icm.pth", map_location=self.device))
        checkpoint = torch.load(f"{path_prefix}_agent_state.pth", map_location=self.device)
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']