import gymnasium as gym
from gymnasium.spaces import Box, MultiDiscrete
import numpy as np
from typing import List, Tuple, Dict, Optional

def generate_terrain(size: int = 12, mountain_prob: float = 0.2, seed: int = None) -> np.ndarray:
    rng = np.random.RandomState(seed)
    terrain = np.zeros((size, size), dtype=np.float32)
    terrain[rng.random((size, size)) < mountain_prob] = 1.0
    return terrain


class Unit:
    def __init__(self, pos: Tuple[int, int], hp: int = 10):
        self.pos: Tuple[int, int] = pos
        self.hp: int = hp


class Board:
    def __init__(self, grid_size: int = 12, seed: Optional[int] = None):

                    self.grid_size: int = grid_size
                    self.max_turns: int = 200
                    self.rng = np.random.RandomState(seed)
                    self.terrain: np.ndarray = generate_terrain(grid_size, 0.2, seed)
                    self.p1_units: List[Unit] = []
                    self.p2_units: List[Unit] = []
                    self.turn: int = 0
                    self.prev_unit_pos: Dict[int, Tuple[int, int]] = {}
                    self.anim_t: float = 0.0
                    self.unit_anims: Dict[int, Tuple[Tuple[int, int], Tuple[int, int]]] = {}
                    self.flash_tiles: set[tuple[int, int]] = set()
                    self.anim_duration: float = 0.3
                    self.place_units(4, 4)

    def is_valid_pos(self, pos: Tuple[int, int]) -> bool:
        r, c = pos
        return (0 <= r < self.grid_size and 0 <= c < self.grid_size and self.terrain[r, c] == 0)

    def is_adjacent(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> bool:
        dr = abs(pos1[0] - pos2[0])
        dc = abs(pos1[1] - pos2[1])
        return max(dr, dc) <= 1

    def get_occupancy(self) -> Dict[Tuple[int, int], List[Unit]]:
        occ: Dict[Tuple[int, int], List[Unit]] = {}
        for unit in self.p1_units + self.p2_units:
            pos = unit.pos
            if pos not in occ:
                occ[pos] = []
            occ[pos].append(unit)
        return occ

    def place_units(self, p1_count: int = 4, p2_count: int = 4) -> None:
        valid_p1 = [(r, c) for r in range(6) for c in range(self.grid_size) if self.is_valid_pos((r, c))]
        self.rng.shuffle(valid_p1)
        p1_pos = valid_p1[:p1_count]
        self.p1_units = [Unit(pos, 10) for pos in p1_pos]

        valid_p2 = [(r, c) for r in range(6, self.grid_size) for c in range(self.grid_size) if self.is_valid_pos((r, c))]
        self.rng.shuffle(valid_p2)
        p2_pos = valid_p2[:p2_count]
        self.p2_units = [Unit(pos, 10) for pos in p2_pos]

    def resolve_turn(self, p1_actions: List[int], p2_actions: Optional[List[int]] = None) -> Dict[str, float]:
            if p2_actions is None:
                p2_actions = [4] * len(self.p2_units)
            deltas = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]
            p1_unit_acts = list(zip(self.p1_units, p1_actions[:len(self.p1_units)]))
            p2_unit_acts = list(zip(self.p2_units, p2_actions[:len(self.p2_units)]))

            intended_targets: Dict[Unit, Tuple[int, int]] = {}
            for unit_acts in [p1_unit_acts, p2_unit_acts]:
                for unit, a in unit_acts:
                    dr, dc = deltas[a]
                    tgt = (unit.pos[0] + dr, unit.pos[1] + dc)
                    intended_targets[unit] = tgt

            # Animation prep
            self.prev_unit_pos = {id(u): u.pos for u in self.p1_units + self.p2_units}
            self.unit_anims.clear()
            self.flash_tiles.clear()

            # Move resolution: priority higher row first
            units_all = self.p1_units + self.p2_units
            sorted_units = sorted(units_all, key=lambda u: (-u.pos[0], u.pos[1]))
            pos_dict = {u: u.pos for u in units_all}
            for u in sorted_units:
                tgt = intended_targets.get(u, u.pos)
                if self.is_valid_pos(tgt):
                    occupied = any(pos_dict[v] == tgt for v in units_all if v != u)
                    if not occupied:
                        u.pos = tgt
                        pos_dict[u] = tgt

            # Attacks
            dmg_p1 = 0.0
            dmg_p2 = 0.0
            attack_damage = 3

            # P1 attacks
            for unit, _ in p1_unit_acts:
                if unit.hp > 0 and unit in self.p1_units:
                    tgt = intended_targets[unit]
                    if self.is_adjacent(unit.pos, tgt) and self.is_valid_pos(tgt):
                        enemies = [u for u in self.p2_units if u.pos == tgt and u.hp > 0]
                        if enemies:
                            enemy = enemies[0]
                            old_hp = enemy.hp
                            enemy.hp = max(0, old_hp - attack_damage)
                            dmg_p2 += old_hp - enemy.hp
                            self.flash_tiles.add(tgt)

            # P2 attacks
            for unit, _ in p2_unit_acts:
                if unit.hp > 0 and unit in self.p2_units:
                    tgt = intended_targets[unit]
                    if self.is_adjacent(unit.pos, tgt) and self.is_valid_pos(tgt):
                        enemies = [u for u in self.p1_units if u.pos == tgt and u.hp > 0]
                        if enemies:
                            enemy = enemies[0]
                            old_hp = enemy.hp
                            enemy.hp = max(0, old_hp - attack_damage)
                            dmg_p1 += old_hp - enemy.hp
                            self.flash_tiles.add(tgt)

            # Remove dead units
            self.p1_units = [u for u in self.p1_units if u.hp > 0]
            self.p2_units = [u for u in self.p2_units if u.hp > 0]

            # Set animations for moved alive units
            all_alive = self.p1_units + self.p2_units
            for u in all_alive:
                uid = id(u)
                if uid in self.prev_unit_pos and self.prev_unit_pos[uid] != u.pos:
                    self.unit_anims[uid] = (self.prev_unit_pos[uid], u.pos)

            self.turn += 1
            self.anim_t = 0.0
            return {'p1_dmg': dmg_p1, 'p2_dmg': dmg_p2}

    def is_done(self) -> Tuple[bool, Optional[str]]:
        if len(self.p1_units) == 0:
            return True, 'p2_win'
        if len(self.p2_units) == 0:
            return True, 'p1_win'
        if self.turn > self.max_turns:
            return True, 'draw'
        return False, None

    def get_winner(self) -> Optional[str]:
        done, outcome = self.is_done()
        if done and outcome != 'draw':
            return outcome
        return None

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.rng.seed(seed)
        
        terrain_seed = self.rng.randint(0, 2**31 - 1)
        self.terrain = generate_terrain(self.grid_size, 0.2, terrain_seed)

        self.p1_units = []
        self.p2_units = []
        self.turn = 0
        self.place_units(4, 4)
        self.anim_t = 0.0
        self.unit_anims.clear()
        self.flash_tiles.clear()

    def compute_hp_density(self, units: List[Unit]) -> np.ndarray:
        density = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        for unit in units:
            r, c = unit.pos
            density[r, c] = max(density[r, c], unit.hp / 10.0)
        return density

    def get_obs(self, view_as_p1: bool = True) -> np.ndarray:
        if view_as_p1:
            ch1 = self.compute_hp_density(self.p1_units)
            ch2 = self.compute_hp_density(self.p2_units)
        else:
            ch1 = self.compute_hp_density(self.p2_units)
            ch2 = self.compute_hp_density(self.p1_units)
        return np.stack([self.terrain, ch1, ch2], axis=-1)


class WarGameEnv(gym.Env):
    def __init__(self, seed: Optional[int] = None):
        super().__init__()
        self.board = Board(seed=seed)
        self.action_space = MultiDiscrete([9] * 4)
        self.observation_space = Box(low=0.0, high=1.0, shape=(12, 12, 3), dtype=np.float32)
        self.opponent_policy = None

    def set_opponent_policy(self, policy) -> None:
        self.opponent_policy = policy

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self.board.reset(seed)
        self.prev_dist = self._get_avg_dist()
        obs = self.board.get_obs(view_as_p1=True)
        return obs, {}

    def _get_avg_dist(self):
        if not self.board.p1_units or not self.board.p2_units:
            return 0
        p1_avg_pos = np.mean([u.pos for u in self.board.p1_units], axis=0)
        p2_avg_pos = np.mean([u.pos for u in self.board.p2_units], axis=0)
        return np.linalg.norm(p1_avg_pos - p2_avg_pos)

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        action_p1 = action.astype(np.int32).tolist()
        if self.opponent_policy is not None:
            obs_p2 = self.board.get_obs(view_as_p1=False)
            action_p2, _ = self.opponent_policy.predict(obs_p2, deterministic=True)
            action_p2 = action_p2.astype(np.int32).tolist()
        else:
            action_p2 = [4] * len(self.board.p2_units)

        damages = self.board.resolve_turn(action_p1, action_p2)
        
        current_dist = self._get_avg_dist()
        dist_delta = self.prev_dist - current_dist
        self.prev_dist = current_dist
        
        reward = 0.001 * damages['p2_dmg'] - 0.001 * damages['p1_dmg'] - 0.001 + 2.5 * dist_delta

        terminated, winner = self.board.is_done()
        truncated = self.board.turn > self.board.max_turns
        if terminated:
            if winner == 'p1_win':
                reward += 10
            elif winner == 'p2_win':
                reward -= 10
        elif truncated:
            p1_hp = sum(u.hp for u in self.board.p1_units)
            p2_hp = sum(u.hp for u in self.board.p2_units)
            reward += 5 * (p1_hp - p2_hp)

        obs = self.board.get_obs(view_as_p1=True)
        info = {'winner': winner}
        return obs, reward, terminated, truncated, info