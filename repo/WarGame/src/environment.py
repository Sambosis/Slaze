"""
WarGame Environment - Core Game Logic
=====================================
Implements the Gymnasium-compatible environment for the tactical wargame.
Handles grid-based physics, unit management, state observation encoding,
and reward computation. Strictly headless - no rendering dependencies.
"""

import gymnasium
from gymnasium import spaces
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
import random
import config
from dataclasses import dataclass


@dataclass
class Unit:
    """Represents a single military unit on the grid."""
    x: int
    y: int
    hp: float
    max_hp: float
    unit_type: str  # 'tank' or 'artillery'
    
    def is_alive(self) -> bool:
        """Check if unit is still operational."""
        return self.hp > 0.0
    
    @property
    def hp_normalized(self) -> float:
        """Normalized HP value for observation encoding [0,1]."""
        return max(0.0, self.hp / self.max_hp)


class WarGameEnv(gymnasium.Env):
    """
    Gymnasium-compatible environment for competitive tank warfare.
    
    State: 7-channel spatial tensor (C,H,W):
        0: Terrain (0=open, 1=obstacle)
        1: Ally positions (1=occupied)
        2: Enemy positions (1=occupied)
        3: Ally unit type (0=none, 0.5=tank, 1.0=artillery)
        4: Enemy unit type (same encoding)
        5: Ally HP normalized [0,1]
        6: Enemy HP normalized [0,1]
    
    Action: Dict mapping unit_id -> action_idx (0-5)
        0: Move Up, 1: Move Down, 2: Move Left, 3: Move Right
        4: Attack Nearest Enemy, 5: Idle
    """
    
    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        
        self.grid_height = config.CFG.game.GRID_HEIGHT
        self.grid_width = config.CFG.game.GRID_WIDTH
        self.max_steps = config.CFG.game.MAX_STEPS
        
        # Unit stats lookup
        self.unit_stats = {
            'tank': {
                'hp': config.CFG.game.TANK_HP,
                'range': config.CFG.game.TANK_RANGE,
                'damage': config.CFG.game.TANK_DAMAGE
            },
            'artillery': {
                'hp': config.CFG.game.ARTILLERY_HP,
                'range': config.CFG.game.ARTILLERY_RANGE,
                'damage': config.CFG.game.ARTILLERY_DAMAGE
            }
        }
        
        # Observation space: (channels, height, width)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(config.CFG.game.OBSERVATION_CHANNELS, self.grid_height, self.grid_width),
            dtype=np.float32
        )
        
        # Action space: Dict of unit_id -> Discrete(6)
        self.action_space = spaces.Dict({
            f"unit_{i}": spaces.Discrete(config.CFG.game.ACTION_SPACE_SIZE)
            for i in range(config.CFG.game.TANK_COUNT + config.CFG.game.ARTILLERY_COUNT)
        })
        
        self.reset()
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Generate random terrain with obstacles
        self.terrain = self._generate_terrain()
        
        # Create units for both players
        self.units_player_a = self._spawn_units('A')
        self.units_player_b = self._spawn_units('B')
        
        self.current_step = 0
        self.done = False
        
        obs = self._get_observation('A')
        info = self._get_info()
        
        return obs, info
    
    def _generate_terrain(self) -> np.ndarray:
        """Procedurally generate terrain with random obstacles."""
        terrain = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)
        
        # Add random obstacles
        for _ in range(int(self.grid_height * self.grid_width * config.CFG.game.WALL_DENSITY)):
            x, y = random.randint(0, self.grid_width-1), random.randint(0, self.grid_height-1)
            terrain[y, x] = 1.0  # Obstacle
        
        # Ensure spawn areas are clear (top-left for A, bottom-right for B)
        spawn_a_clear = [(2,2), (2,3), (3,2), (4,4)]
        spawn_b_clear = [(self.grid_height-3, self.grid_width-3),
                        (self.grid_height-3, self.grid_width-4),
                        (self.grid_height-4, self.grid_width-3),
                        (self.grid_height-5, self.grid_width-5)]
        
        for x, y in spawn_a_clear + spawn_b_clear:
            terrain[y, x] = 0.0
        
        return terrain
    
    def _spawn_units(self, player: str) -> List[Unit]:
        """Spawn units for a player in a safe starting position."""
        unit_types = (['tank'] * config.CFG.game.TANK_COUNT + 
                     ['artillery'] * config.CFG.game.ARTILLERY_COUNT)
        
        if player == 'A':
            positions = [(2,2), (2,3), (3,2), (4,4)]
        else:  # Player B
            positions = [(self.grid_height-3, self.grid_width-3),
                        (self.grid_height-3, self.grid_width-4),
                        (self.grid_height-4, self.grid_width-3),
                        (self.grid_height-5, self.grid_width-5)]
        
        units = []
        for i, (utype, (x, y)) in enumerate(zip(unit_types, positions)):
            stats = self.unit_stats[utype]
            units.append(Unit(x=x, y=y, hp=stats['hp'], max_hp=stats['hp'], unit_type=utype))
        
        return units
    
    def step(self, action_dict: Dict[str, int]) -> Tuple[np.ndarray, float, bool, bool, Dict]:
                if self.done:
                    raise RuntimeError("Environment is done, call reset()")
        
                # Separate actions for Player A (keys: 'unit_i') and Player B (keys: 'opponent_i')
                a_actions = {k: v for k, v in action_dict.items() if not k.startswith("opponent_")}
        
                # Transform 'opponent_i' keys to 'unit_i' for internal processing of Player B
                b_actions = {k.replace("opponent_", "unit_"): v 
                             for k, v in action_dict.items() if k.startswith("opponent_")}
        
                # Execute actions for player A
                self._execute_actions(self.units_player_a, a_actions, is_player_a=True)
        
                # Execute actions for player B
                # Fallback to random if no specific actions provided (e.g. legacy mode)
                if not b_actions:
                    b_actions = {f"unit_{i}": random.randint(0, 5) 
                                 for i in range(len(self.units_player_b))}
        
                self._execute_actions(self.units_player_b, b_actions, is_player_a=False)
        
                # Resolve attacks and combat
                self._resolve_attacks()
        
                self.current_step += 1
        
                # Check termination conditions
                truncated = self.current_step >= self.max_steps
                terminated = self._is_terminal()
                self.done = terminated or truncated
        
                reward = self._compute_reward()
                obs = self._get_observation('A')
                info = self._get_info()
        
                return obs, reward, terminated, truncated, info
    
    def _execute_actions(self, units: List[Unit], action_dict: Dict[str, int], 
                        is_player_a: bool) -> None:
        """Process movement and attack actions for a player's units."""
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
        
        for i, unit in enumerate(units):
            if not unit.is_alive():
                continue
                
            action = action_dict.get(f"unit_{i}", 5)  # Default to idle
            
            if action < 4:  # Movement
                dx, dy = directions[action]
                new_x, new_y = unit.x + dx, unit.y + dy
                
                # Check bounds and collision
                if (0 <= new_x < self.grid_width and 
                    0 <= new_y < self.grid_height and 
                    self.terrain[new_y, new_x] == 0 and
                    not self._is_position_occupied(new_x, new_y, is_player_a)):
                    
                    unit.x, unit.y = new_x, new_y
            
            # Attack action (4) handled in _resolve_attacks()
    
    def _resolve_attacks(self) -> None:
        """Resolve all attack actions - find targets and apply damage."""
        for unit_a in self.units_player_a:
            if not unit_a.is_alive():
                continue
            
            # Find nearest enemy in range
            target = self._find_nearest_enemy(unit_a, self.units_player_b)
            if target:
                distance = self._chebyshev_distance(unit_a.x, unit_a.y, target.x, target.y)
                if distance <= self.unit_stats[unit_a.unit_type]['range']:
                    damage = self.unit_stats[unit_a.unit_type]['damage']
                    target.hp = max(0.0, target.hp - damage)
        
        # Player B attacks
        for unit_b in self.units_player_b:
            if not unit_b.is_alive():
                continue
            
            target = self._find_nearest_enemy(unit_b, self.units_player_a)
            if target:
                distance = self._chebyshev_distance(unit_b.x, unit_b.y, target.x, target.y)
                if distance <= self.unit_stats[unit_b.unit_type]['range']:
                    damage = self.unit_stats[unit_b.unit_type]['damage']
                    target.hp = max(0.0, target.hp - damage)
    
    def _find_nearest_enemy(self, unit: Unit, enemy_units: List[Unit]) -> Optional[Unit]:
        """Find nearest living enemy using Chebyshev distance."""
        nearest = None
        min_dist = float('inf')
        
        for enemy in enemy_units:
            if enemy.is_alive():
                dist = self._chebyshev_distance(unit.x, unit.y, enemy.x, enemy.y)
                if dist < min_dist:
                    min_dist = dist
                    nearest = enemy
        
        return nearest
    
    def _chebyshev_distance(self, x1: int, y1: int, x2: int, y2: int) -> int:
        """Chebyshev distance (max(dx, dy)) for attack range."""
        return max(abs(x1 - x2), abs(y1 - y2))
    
    def _is_position_occupied(self, x: int, y: int, is_player_a: bool) -> bool:
        """Check if position is occupied by any unit."""
        all_units = self.units_player_a if is_player_a else self.units_player_b
        opponent_units = self.units_player_b if is_player_a else self.units_player_a
        
        # Check friendly units
        for unit in all_units:
            if unit.is_alive() and unit.x == x and unit.y == y:
                return True
        
        # No friendly fire check for movement (units can stack minimally)
        return False
    
    def _is_terminal(self) -> bool:
        """Check if game is over (one side eliminated)."""
        player_a_alive = any(unit.is_alive() for unit in self.units_player_a)
        player_b_alive = any(unit.is_alive() for unit in self.units_player_b)
        
        return not player_a_alive or not player_b_alive
    
    def _compute_reward(self) -> float:
        """Compute dense reward signal for RL training."""
        if self.done:
            # Win/loss bonus
            a_alive = sum(unit.is_alive() for unit in self.units_player_a)
            b_alive = sum(unit.is_alive() for unit in self.units_player_b)
            if a_alive > 0 and b_alive == 0:
                return 10.0  # Win
            elif a_alive == 0:
                return -10.0  # Loss
            return 0.0
        
        # Dense shaping: HP advantage + enemy kills
        a_hp_total = sum(unit.hp for unit in self.units_player_a if unit.is_alive())
        b_hp_total = sum(unit.hp for unit in self.units_player_b if unit.is_alive())
        
        hp_advantage = (a_hp_total - b_hp_total) / 400.0  # Normalize
        
        return hp_advantage * 0.1
    
    def _get_observation(self, player: str) -> np.ndarray:
        """Generate 7-channel observation tensor for the given player."""
        obs = np.zeros((
            config.CFG.game.OBSERVATION_CHANNELS,
            self.grid_height,
            self.grid_width
        ), dtype=np.float32)
        
        # Channel 0: Terrain
        obs[0] = self.terrain
        
        # Determine ally/enemy units
        ally_units = self.units_player_a if player == 'A' else self.units_player_b
        enemy_units = self.units_player_b if player == 'A' else self.units_player_a
        
        # Channel 1: Ally positions
        # Channel 3: Ally unit type
        # Channel 5: Ally HP
        for unit in ally_units:
            if unit.is_alive():
                obs[1, unit.y, unit.x] = 1.0
                obs[3, unit.y, unit.x] = 0.5 if unit.unit_type == 'tank' else 1.0
                obs[5, unit.y, unit.x] = unit.hp_normalized
        
        # Channel 2: Enemy positions
        # Channel 4: Enemy unit type
        # Channel 6: Enemy HP
        for unit in enemy_units:
            if unit.is_alive():
                obs[2, unit.y, unit.x] = 1.0
                obs[4, unit.y, unit.x] = 0.5 if unit.unit_type == 'tank' else 1.0
                obs[6, unit.y, unit.x] = unit.hp_normalized
        
        return obs
    
    def _get_info(self) -> Dict:
        """Return auxiliary diagnostic information."""
        a_alive = sum(1 for u in self.units_player_a if u.is_alive())
        b_alive = sum(1 for u in self.units_player_b if u.is_alive())
        a_hp = sum(u.hp for u in self.units_player_a if u.is_alive())
        b_hp = sum(u.hp for u in self.units_player_b if u.is_alive())
        
        return {
            'step': self.current_step,
            'player_a_alive': a_alive,
            'player_b_alive': b_alive,
            'player_a_hp': a_hp,
            'player_b_hp': b_hp
        }
    
    def get_state_tensor(self, player: str = 'A') -> torch.Tensor:
        """Convenience method to get PyTorch observation tensor."""
        obs = self._get_observation(player)
        return torch.from_numpy(obs).float().unsqueeze(0)  # Add batch dim
    
    @property
    def player_a_won(self) -> bool:
        """Check if player A achieved victory."""
        return self.done and any(u.is_alive() for u in self.units_player_a)