import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config

class DQN(nn.Module):
    """
    Double Deep Q-Network (DDQN) for processing grid-based game states.
    
    Input: State tensor of shape (batch_size, channels=5, grid_size=16, grid_size=16)
    Output: Q-values for each action (batch_size, n_actions=6)
    
    Architecture:
    - Convolutional layers to extract spatial features from grid
    - Flatten and fully connected layers to produce action Q-values
    """
    def __init__(self, config: Config, n_actions: int = None):
        super(DQN, self).__init__()
        self.config = config
        self.grid_size = config.GRID_SIZE
        self.state_channels = config.STATE_CHANNELS
        self.n_actions = n_actions or config.N_ACTIONS
        
        # Convolutional backbone
        self.conv_layers = nn.Sequential(
            # Input: (B, 5, 16, 16)
            nn.Conv2d(self.state_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        
        # Calculate flattened size after conv layers
        with torch.no_grad():
            sample_input = torch.zeros(1, self.state_channels, self.grid_size, self.grid_size)
            conv_out = self.conv_layers(sample_input)
            self.flattened_size = conv_out.view(1, -1).shape[1]
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(self.flattened_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.n_actions)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through DQN.
        
        Args:
            state: (batch_size, channels, grid_size, grid_size)
        
        Returns:
            q_values: (batch_size, n_actions)
        """
        # Ensure input is float and proper shape
        if state.dtype != torch.float32:
            state = state.float()
        
        # Conv layers
        x = self.conv_layers(state)
        # Flatten
        x = x.view(x.size(0), -1)
        # FC layers
        q_values = self.fc_layers(x)
        
        return q_values


class ICMModel(nn.Module):
    """
    Intrinsic Curiosity Module (ICM) for curiosity-driven exploration.
    
    Components:
    1. feature_encoder: Encodes raw state to latent features (shared)
    2. inverse_model: Predicts action from (state, next_state) features
    3. forward_model: Predicts next_state features from (state features, action)
    
    Intrinsic reward = ||forward_model(state_feat, action) - next_state_feat||^2
    
    Args:
        config: Global configuration
        feature_dim: Size of encoded feature space (default: 256)
        n_actions: Number of discrete actions (default: 6)
    """
    def __init__(self, config: Config, feature_dim: int = None, n_actions: int = None):
        super(ICMModel, self).__init__()
        self.config = config
        self.feature_dim = feature_dim or config.ICM_FEATURE_DIM
        self.n_actions = n_actions or config.N_ACTIONS
        
        self.grid_size = config.GRID_SIZE
        self.state_channels = config.STATE_CHANNELS
        
        # 1. Shared feature encoder (same as DQN conv backbone but smaller)
        self.feature_encoder = nn.Sequential(
            nn.Conv2d(self.state_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        
        # Calculate feature size
        with torch.no_grad():
            sample_input = torch.zeros(1, self.state_channels, self.grid_size, self.grid_size)
            feat_out = self.feature_encoder(sample_input)
            self.raw_feature_size = feat_out.view(1, -1).shape[1]
        
        self.feature_projection = nn.Linear(self.raw_feature_size, self.feature_dim)
        
        # 2. Inverse model: Predicts action from (state_feat, next_state_feat)
        self.inverse_model = nn.Sequential(
            nn.Linear(self.feature_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.n_actions)
        )
        
        # 3. Forward model: Predicts next_state_feat from (state_feat, one_hot_action)
        self.forward_model = nn.Sequential(
            nn.Linear(self.feature_dim + self.n_actions, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.feature_dim)
        )
    
    def encode_state(self, state: torch.Tensor) -> torch.Tensor:
        """
        Encode raw state tensor to feature vector.
        
        Args:
            state: (batch_size, channels, H, W)
        
        Returns:
            features: (batch_size, feature_dim)
        """
        if state.dtype != torch.float32:
            state = state.float()
        
        x = self.feature_encoder(state)
        x = x.view(x.size(0), -1)
        features = self.feature_projection(x)
        return features
    
    def forward(self, state: torch.Tensor, next_state: torch.Tensor, action: torch.Tensor) -> dict:
        """
        Full ICM forward pass.
        
        Args:
            state: (batch_size, channels, H, W)
            next_state: (batch_size, channels, H, W) 
            action: (batch_size,) or (batch_size, 1) action indices
        
        Returns:
            dict containing:
                - 'state_feat': (batch_size, feature_dim)
                - 'next_state_feat': (batch_size, feature_dim)
                - 'pred_action': (batch_size, n_actions) - inverse model logits
                - 'pred_next_feat': (batch_size, feature_dim) - forward model prediction
        """
        batch_size = state.shape[0]
        
        # Encode states
        state_feat = self.encode_state(state)
        next_state_feat = self.encode_state(next_state)
        
        # Inverse model: predict action
        inverse_input = torch.cat([state_feat, next_state_feat], dim=1)
        pred_action = self.inverse_model(inverse_input)
        
        # Forward model: predict next features
        # Convert action to one-hot
        if action.dim() == 1:
            action_onehot = F.one_hot(action.long(), num_classes=self.n_actions).float()
        else:
            action_onehot = action.float()
        
        forward_input = torch.cat([state_feat, action_onehot], dim=1)
        pred_next_feat = self.forward_model(forward_input)
        
        return {
            'state_feat': state_feat,
            'next_state_feat': next_state_feat,
            'pred_action': pred_action,
            'pred_next_feat': pred_next_feat
        }
    
    def compute_intrinsic_reward(self, state_feat: torch.Tensor, next_state_feat: torch.Tensor, 
                               action: torch.Tensor) -> torch.Tensor:
        """
        Compute intrinsic (curiosity) reward from forward model prediction error.
        
        Args:
            state_feat: (batch_size, feature_dim)
            next_state_feat: (batch_size, feature_dim) - ground truth
            action: (batch_size,) action indices
            
        Returns:
            intrinsic_reward: (batch_size,) prediction error (L2 norm)
        """
        if action.dim() == 1:
            action_onehot = F.one_hot(action.long(), num_classes=self.n_actions).float()
        else:
            action_onehot = action.float()
            
        forward_input = torch.cat([state_feat, action_onehot], dim=1)
        pred_next_feat = self.forward_model(forward_input)
        
        # L2 prediction error (intrinsic reward)
        intrinsic_reward = F.mse_loss(pred_next_feat, next_state_feat, reduction='none').mean(dim=1)
        return intrinsic_reward