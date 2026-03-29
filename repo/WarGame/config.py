import os
import torch
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

@dataclass
class PathConfig:
    """File system paths and directory management."""
    BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
    CHECKPOINT_DIR: str = os.path.join(BASE_DIR, "checkpoints")
    HALL_OF_FAME_DIR: str = os.path.join(CHECKPOINT_DIR, "hall_of_fame")
    LOG_DIR: str = os.path.join(BASE_DIR, "logs")
    ASSETS_DIR: str = os.path.join(BASE_DIR, "assets")

    def create_dirs(self) -> None:
        """Ensure all necessary directories exist."""
        os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(self.HALL_OF_FAME_DIR, exist_ok=True)
        os.makedirs(self.LOG_DIR, exist_ok=True)
        os.makedirs(self.ASSETS_DIR, exist_ok=True)

@dataclass
class GameConfig:
    """Game rules, grid dimensions, and unit statistics."""
    # Grid Dimensions
    GRID_HEIGHT: int = 24
    GRID_WIDTH: int = 32
    CELL_SIZE: int = 24  # Pixels per cell for visualization

    # Episode Constraints
    MAX_STEPS: int = 1024  # Max steps before truncation
    
    # Unit Composition (Per Agent)
    TANK_COUNT: int = 3
    ARTILLERY_COUNT: int = 1
    
    # Unit Stats: Tank (Brawler)
    TANK_HP: float = 100.0
    TANK_RANGE: int = 3      # Chebyshev distance
    TANK_DAMAGE: float = 25.0
    
    # Unit Stats: Artillery (Glass Cannon)
    ARTILLERY_HP: float = 50.0
    ARTILLERY_RANGE: int = 8 # Chebyshev distance
    ARTILLERY_DAMAGE: float = 40.0
    
    # Environment Logic
    FRIENDLY_FIRE: bool = False
    WALL_DENSITY: float = 0.15  # Probability of obstacle generation
    
    # State Representation (Channels for CNN)
    # 0: Terrain (0=Open, 1=Obstacle)
    # 1: Ally Unit Position
    # 2: Enemy Unit Position
    # 3: Ally Unit Type (0=None, 0.5=Tank, 1.0=Artillery)
    # 4: Enemy Unit Type
    # 5: Ally HP (Normalized)
    # 6: Enemy HP (Normalized)
    OBSERVATION_CHANNELS: int = 7
    
    # Action Space
    # 0: Up, 1: Down, 2: Left, 3: Right, 4: Attack Nearest, 5: Idle
    ACTION_SPACE_SIZE: int = 6

@dataclass
class ModelConfig:
    """Neural Network Architecture Hyperparameters."""
    # CNN Architecture
    CNN_CHANNELS: List[int] = field(default_factory=lambda: [32, 64, 128])
    CNN_KERNEL_SIZE: int = 3
    CNN_STRIDE: int = 1
    CNN_PADDING: int = 1
    
    # Feature Extraction
    EMBEDDING_SIZE: int = 256
    
    # Actor-Critic Heads
    ACTOR_HIDDEN_SIZES: List[int] = field(default_factory=lambda: [256, 128])
    CRITIC_HIDDEN_SIZES: List[int] = field(default_factory=lambda: [256, 128])
    
    # Initialization
    ORTHOGONAL_INIT: bool = True

@dataclass
class TrainingConfig:
    """PPO and Self-Play Hyperparameters."""
    # Optimizer
    LEARNING_RATE: float = 3e-4
    ANNEAL_LR: bool = True
    MAX_GRAD_NORM: float = 0.5
    
    # PPO Parameters
    GAMMA: float = 0.99             # Discount factor
    GAE_LAMBDA: float = 0.95        # GAE smoothing
    CLIP_COEF: float = 0.2          # PPO clip parameter
    ENTROPY_COEF: float = 0.01      # Exploration incentive
    VALUE_COEF: float = 0.5         # Critic loss weight
    
    # Training Loop
    TOTAL_TIMESTEPS: int = 2_000_000
    NUM_ENVS: int = 1               # Number of parallel environments (if vectorized)
    STEPS_PER_BATCH: int = 2048     # Steps collected before update
    NUM_MINIBATCHES: int = 32
    UPDATE_EPOCHS: int = 4
    
    # Hall of Fame / Self-Play
    OPPONENT_LATEST_PROB: float = 0.8  # 80% vs latest self, 20% vs history
    CHECKPOINT_INTERVAL: int = 1000    # Save model to history every N updates
    
    # Device
    DEVICE: str = "auto"  # "auto", "cuda", "mps", "cpu"

    @property
    def batch_size(self) -> int:
        return self.STEPS_PER_BATCH * self.NUM_ENVS

@dataclass
class VisualizerConfig:
    """Pygame Rendering Settings."""
    # Window
    SCREEN_WIDTH: int = 1024
    SCREEN_HEIGHT: int = 768
    FPS_TRAINING: int = 0    # 0 = Uncapped
    FPS_SPECTATOR: int = 30  # Human watchable speed
    
    # Toggles
    RENDER_FREQUENCY: int = 500  # Render a match every N updates
    
    # Colors (R, G, B)
    COLOR_BG: Tuple[int, int, int] = (20, 20, 30)
    COLOR_GRID: Tuple[int, int, int] = (40, 40, 50)
    COLOR_OBSTACLE: Tuple[int, int, int] = (80, 80, 90)
    
    COLOR_AGENT_A: Tuple[int, int, int] = (65, 105, 225)  # Royal Blue
    COLOR_AGENT_B: Tuple[int, int, int] = (220, 20, 60)   # Crimson
    
    COLOR_HP_FULL: Tuple[int, int, int] = (0, 255, 0)
    COLOR_HP_LOW: Tuple[int, int, int] = (255, 0, 0)
    COLOR_TEXT: Tuple[int, int, int] = (240, 240, 240)

@dataclass
class Config:
    """Master Configuration Object."""
    paths: PathConfig = field(default_factory=PathConfig)
    game: GameConfig = field(default_factory=GameConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    view: VisualizerConfig = field(default_factory=VisualizerConfig)

    def get_device(self) -> torch.device:
        """Resolves the computing device based on availability and config."""
        if self.training.DEVICE != "auto":
            return torch.device(self.training.DEVICE)
        
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

# Global Configuration Instance
CFG = Config()

# Ensure directories exist upon import
CFG.paths.create_dirs()