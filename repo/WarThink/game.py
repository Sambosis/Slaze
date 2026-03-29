from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional, Dict

class Unit:
    """
    Represents a single army unit with position, stats, and action methods.
    
    Attributes:
        pos: Current grid position (row, col)
        type_: Unit type ('warrior', 'archer', 'commander')
        hp: Current hit points
        max_hp: Maximum hit points
        player: Owning player (1 or 2)
        alive: Whether unit is alive
    """
    def __init__(self, pos: Tuple[int, int], type_: str, player: int):
        if type_ not in ('warrior', 'archer', 'commander'):
            raise ValueError(f'Invalid unit type: {type_}')
        self.pos = pos
        self.type_ = type_
        self.player = player
        self.max_hp = 5 if type_ == 'commander' else 3
        self.hp = self.max_hp
        self.alive = True
    
    def move(self, new_pos: Tuple[int, int]):
        """Move unit to new grid position (with bounds checking done externally)."""
        self.pos = new_pos
    
    def attack(self, target: Unit):
        """Deal damage to target unit."""
        damage = 2 if self.type_ == 'commander' else 1
        target.hp = max(0, target.hp - damage)
        if target.hp == 0:
            target.alive = False

class GameState:
    """
    Manages the full game board, units, and state transitions for a single episode.
    
    Attributes:
        grid: Observation tensor (10,10,5) - updated by generate_obs
        p1_units: List of Player 1 units
        p2_units: List of Player 2 units
        current_player: Active player (1 or 2)
        turn_count: Number of completed turns
        winner: Winner (1, 2, or 0 for tie) or None
    """
    def __init__(self):
        self.grid = np.zeros((10, 10, 5), dtype=np.float32)
        self.p1_units: List[Unit] = []
        self.p2_units: List[Unit] = []
        self.current_player = 1
        self.turn_count = 0
        self.winner: Optional[int] = None
        self.win_condition: Optional[str] = None
    
    def reset(self):
        """Reset to initial game state with units in opposite corners."""
        unit_types = ['warrior', 'warrior', 'archer', 'archer', 'commander']
        # P1 starts bottom-left
        p1_positions = [(9, 0), (9, 1), (8, 0), (8, 1), (9, 2)]
        self.p1_units = [Unit(p1_positions[i], unit_types[i], 1) for i in range(5)]
        # P2 starts top-right
        p2_positions = [(0, 9), (1, 9), (0, 8), (1, 8), (0, 7)]
        self.p2_units = [Unit(p2_positions[i], unit_types[i], 2) for i in range(5)]
        self.current_player = 1
        self.turn_count = 0
        self.winner = None
        self.win_condition = None
        self.grid = np.zeros((10, 10, 5), dtype=np.float32)
    
    def get_current_units(self) -> List[Unit]:
        """Get units for the current player."""
        return self.p1_units if self.current_player == 1 else self.p2_units
    
    def get_enemy_units(self) -> List[Unit]:
        """Get units for the enemy player."""
        return self.p2_units if self.current_player == 1 else self.p1_units
    
    def is_done(self) -> bool:
        """Check victory/draw conditions and set win_condition."""
        p1_alive_units = [u for u in self.p1_units if u.alive]
        p2_alive_units = [u for u in self.p2_units if u.alive]

        if not p1_alive_units:
            self.winner = 2
            self.win_condition = 'annihilation'
            return True
        if not p2_alive_units:
            self.winner = 1
            self.win_condition = 'annihilation'
            return True
        
        if self.turn_count >= 200:
            p1_count = len(p1_alive_units)
            p2_count = len(p2_alive_units)
            if p1_count > p2_count:
                self.winner = 1
                self.win_condition = 'attrition'
            elif p2_count > p1_count:
                self.winner = 2
                self.win_condition = 'attrition'
            else:
                self.winner = 0 # True tie
                self.win_condition = 'draw'
            return True
            
        return False

def generate_obs(state: GameState) -> np.ndarray:
    obs = np.zeros((10, 10, 5), dtype=np.float32)
    type_map = {'warrior': 1.0, 'archer': 2.0, 'commander': 3.0}

    # P1 units type/HP
    for unit in state.p1_units:
        if unit.alive:
            y, x = unit.pos
            obs[y, x, 1] = type_map[unit.type_] / 3.0  # P1 type
            obs[y, x, 2] = unit.hp / unit.max_hp  # P1 HP

    # P2 units type/HP
    for unit in state.p2_units:
        if unit.alive:
            y, x = unit.pos
            obs[y, x, 3] = type_map[unit.type_] / 3.0  # P2 type
            obs[y, x, 4] = unit.hp / unit.max_hp  # P2 HP base

    # Turn indicator ch0
    if state.current_player == 1:
        obs[:, :, 0] = 1.0

    # Valid moves mask ch4 (max with P2 HP)
    current_units = state.get_current_units()
    deltas = [(-1, 0), (1, 0), (0, 1), (0, -1)]  # N S E W
    mask_grid = np.zeros((10, 10), dtype=np.float32)
    for unit in current_units:
        if not unit.alive:
            continue
        y, x = unit.pos
        mask_grid[y, x] = 1.0  # stay pos

        move_range = 2 if unit.type_ == 'commander' else 1
        attack_range = 3 if unit.type_ == 'archer' else 1

        # move targets
        for dy, dx in deltas:
            ny = max(0, min(9, y + dy * move_range))
            nx = max(0, min(9, x + dx * move_range))
            mask_grid[ny, nx] = 1.0

        # attack ray targets
        for dy, dx in deltas:
            for step in range(1, attack_range + 1):
                ny = max(0, min(9, y + dy * step))
                nx = max(0, min(9, x + dx * step))
                mask_grid[ny, nx] = 1.0

    obs[:, :, 4] = np.maximum(obs[:, :, 4], mask_grid)

    state.grid = obs.copy()
    return obs

def resolve_actions(state: GameState, actions: np.ndarray) -> List[Tuple[str, str]]:
    if actions.shape != (5,):
        raise ValueError(f"Actions must be shape (5,), got {actions.shape}")

    current_units = state.get_current_units()
    enemy_units = state.get_enemy_units()
    logs: List[Tuple[str, str]] = []
    deltas = [(0, 0), (-1, 0), (1, 0), (0, 1), (0, -1)]  # stay, N, S, E, W
    dir_labels = ['', 'N', 'S', 'E', 'W']  # index 1-4
    intended_moves: Dict[Tuple[int, int], List[int]] = {}
    attack_list: List[Tuple[int, int, int, int, int]] = []  # unit_idx, dy, dx, max_range, log_idx

    current_player_str = str(state.current_player)

    # Parse actions
    for i in range(5):
        unit_id = f'{current_player_str}u{i}'
        if not current_units[i].alive:
            logs.append((unit_id, 'stay (dead)'))
            continue
    
        act = int(actions[i])
        unit = current_units[i]
    
        if act < 5:  # move/stay
            dy, dx = deltas[act]
            move_range = 2 if unit.type_ == 'commander' else 1
            new_y = max(0, min(9, unit.pos[0] + dy * move_range))
            new_x = max(0, min(9, unit.pos[1] + dx * move_range))
            new_pos = (new_y, new_x)
            intended_moves.setdefault(new_pos, []).append(i)
            logs.append((unit_id, f'move to ({new_y}, {new_x})'))
        else:  # attack
            dir_idx = act - 5
            dy, dx = deltas[dir_idx + 1]
            attack_range = 3 if unit.type_ == 'archer' else 1
            dir_label = dir_labels[dir_idx + 1]
            logs.append((unit_id, f'attack {dir_label}'))
            log_idx = len(logs) - 1
            attack_list.append((i, dy, dx, attack_range, log_idx))

    # Resolve moves: priority to single claimants, skip collisions
    for target_pos, claimants in intended_moves.items():
        if len(claimants) == 1:
            current_units[claimants[0]].move(target_pos)

    # Resolve attacks: nearest enemy on ray in direction up to range
    for unit_idx, dy, dx, max_range, log_idx in attack_list:
        unit = current_units[unit_idx]
        y_pos, x_pos = unit.pos
        target = None
        for step in range(1, max_range + 1):
            target_y = max(0, min(9, y_pos + dy * step))
            target_x = max(0, min(9, x_pos + dx * step))
            target_pos = (target_y, target_x)
            for enemy in enemy_units:
                if enemy.alive and enemy.pos == target_pos:
                    target = enemy
                    break
            if target:
                break
        if target:
            unit.attack(target)
            enemy_idx = enemy_units.index(target)
            old_unit_id, _ = logs[log_idx]
            logs[log_idx] = (old_unit_id, f'attack hit enemy u{enemy_idx}')

    return logs
