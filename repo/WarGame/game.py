from typing import List
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict
from collections import defaultdict


@dataclass
class Unit:
    slot_id: int
    pos: Tuple[int, int]
    hp: int
    max_hp: int
    atk: int
    atk_range: int
    speed: int
    type: str
    owner: str

    def take_damage(self, dmg: int) -> bool:
        """Apply damage and return True if killed (hp <= 0)."""
        self.hp -= dmg
        return self.hp <= 0


class GameState:
    def __init__(self):
        self.width: int = 15
        self.height: int = 15
        self._generate_terrain()
        self.base_hp_red: int = 100
        self.base_hp_blue: int = 100
        self.bases_red_alive: bool = True
        self.bases_blue_alive: bool = True
        self.resources_red: int = 0
        self.resources_blue: int = 0
        self.turn_count: int = 0
        self._init_units()
        self.dirs_move: List[Tuple[int, int]] = [
            (0, 0),  # stay
            (-1, 0), (-1, 1), (0, 1),
            (1, 1), (1, 0), (1, -1),
            (0, -1), (-1, -1)
        ]
        self.dirs_attack: List[Tuple[int, int]] = [(-1, 0), (1, 0), (0, 1), (0, -1)]  # N S E W
        self.fog_red: np.ndarray = np.zeros((15, 15), dtype=bool)
        self.fog_blue: np.ndarray = np.zeros((15, 15), dtype=bool)
        self.update_fog('red')
        self.update_fog('blue')

    def _generate_terrain(self) -> None:
        """Generate random terrain: 0=grass, 1=forest (+20% def), 2=hills (halve speed, ignored)."""
        self.terrain: np.ndarray = np.random.choice(
            [0, 1, 2], size=(self.height, self.width), p=[0.7, 0.2, 0.1]
        ).astype(np.int32)

    def _init_units(self) -> None:
        """Initialize 5 units per side: slots 0-2 Infantry, 3-4 Tanks."""
        inf_stats = {'hp': 15, 'max_hp': 15, 'atk': 5, 'atk_range': 1, 'speed': 1, 'type': 'inf'}
        tank_stats = {'hp': 30, 'max_hp': 30, 'atk': 10, 'atk_range': 2, 'speed': 2, 'type': 'tank'}
        red_positions = [(0, 0), (0, 1), (1, 0), (1, 1), (0, 2)]
        self.units_red: List[Unit] = [
            Unit(i, red_positions[i], **inf_stats if i < 3 else tank_stats, owner='red')
            for i in range(5)
        ]
        blue_positions = [(14, 14), (14, 13), (13, 14), (13, 13), (14, 12)]
        self.units_blue: List[Unit] = [
            Unit(i, blue_positions[i], **inf_stats if i < 3 else tank_stats, owner='blue')
            for i in range(5)
        ]

    def revive_dead(self, owner: str) -> int:
        """Revive dead units (hp <= 0) to full HP at base pos. Returns num revived."""
        if owner == 'red':
            units = self.units_red
            base_pos = (0, 0)
        else:
            units = self.units_blue
            base_pos = (14, 14)
        revived = 0
        for unit in units:
            if unit.hp <= 0:
                unit.pos = base_pos
                unit.hp = unit.max_hp
                revived += 1
        return revived

    def _compute_fog(self, centers: List[Tuple[int, int]]) -> np.ndarray:
        """Compute fog of war: Chebyshev radius 4 around centers."""
        fog = np.zeros((self.height, self.width), dtype=bool)
        for cr, cc in centers:
            for dr in range(-4, 5):
                for dc in range(-4, 5):
                    if max(abs(dr), abs(dc)) <= 4:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < self.height and 0 <= nc < self.width:
                            fog[nr, nc] = True
        return fog

    def update_fog(self, owner: str) -> np.ndarray:
        """Update fog for owner based on alive units + base if alive."""
        if owner == 'red':
            centers = [u.pos for u in self.units_red if u.hp > 0]
            if self.bases_red_alive:
                centers.append((0, 0))
            self.fog_red = self._compute_fog(centers)
            return self.fog_red
        else:
            centers = [u.pos for u in self.units_blue if u.hp > 0]
            if self.bases_blue_alive:
                centers.append((14, 14))
            self.fog_blue = self._compute_fog(centers)
            return self.fog_blue

    def get_action_meaning(self, action: int) -> Tuple[str, Tuple[int, int]]:
        """Decode action: 0-8 move/stay (9 dirs), 9-12 attack (4 cardinals)."""
        if action < 9:
            return 'move', self.dirs_move[action]
        else:
            return 'attack', self.dirs_attack[action - 9]
    def compute_target_pos(self, pos: Tuple[int, int], dr: int, dc: int, speed: int) -> Tuple[int, int]:
        r, c = pos
        tent_r = max(0, min(self.height - 1, r + dr * speed))
        tent_c = max(0, min(self.width - 1, c + dc * speed))
        terr = self.terrain[tent_r, tent_c]
        eff_speed = speed // 2 if terr == 2 else speed
        nr = max(0, min(self.height - 1, r + dr * eff_speed))
        nc = max(0, min(self.width - 1, c + dc * eff_speed))
        return (nr, nc)
    def _resolve_attacks(self, attackers: List[Unit], actions: List[int], defenders: List[Unit],
                             defender_base_pos: Tuple[int, int], base_alive: bool,
                             unit_dmg: Dict[int, int], base_dmg: List[int]) -> None:
        for i, action in enumerate(actions):
            typ, _ = self.get_action_meaning(action)
            if typ != 'attack':
                continue
            unit = attackers[i]
            dir_idx = action - 9
            dr, dc = self.dirs_attack[dir_idx]
            target_pos = None
            target_slot = None
            is_base = False
            for k in range(1, unit.atk_range + 1):
                tr = unit.pos[0] + k * dr
                tc = unit.pos[1] + k * dc
                if not (0 <= tr < self.height and 0 <= tc < self.width):
                    break
                cand_pos = (tr, tc)
                target_slot_cand = None
                for j, du in enumerate(defenders):
                    if du.pos == cand_pos and du.hp > 0:
                        target_slot_cand = j
                        break
                if target_slot_cand is not None:
                    target_pos = cand_pos
                    target_slot = target_slot_cand
                    is_base = False
                    break
                if cand_pos == defender_base_pos and base_alive:
                    target_pos = cand_pos
                    is_base = True
                    break
            if target_pos is not None:
                terr = self.terrain[target_pos[0], target_pos[1]]
                mult = 0.8 if terr == 1 else 1.0
                dmg = int(unit.atk * mult)
                if is_base:
                    base_dmg[0] += dmg
                else:
                    unit_dmg[target_slot] += dmg
    def step(self, actions_red: List[int], actions_blue: List[int]) -> Tuple[Dict[str, float], bool]:
        """Execute one turn: res, revive, move, attack, rewards, fog, done."""
        self.turn_count += 1
        res_red_gain = 2 if self.bases_red_alive else 0
        self.resources_red += res_red_gain
        res_blue_gain = 2 if self.bases_blue_alive else 0
        self.resources_blue += res_blue_gain
        rev_red = self.revive_dead('red')
        rev_blue = self.revive_dead('blue')
        
        # Intended positions for moves/stay
        intended_red = {}
        for i in range(5):
            typ, delta = self.get_action_meaning(actions_red[i])
            intended_red[i] = (
                self.compute_target_pos(self.units_red[i].pos, *delta, self.units_red[i].speed)
                if typ == 'move' else self.units_red[i].pos
            )
        intended_blue = {}
        for i in range(5):
            typ, delta = self.get_action_meaning(actions_blue[i])
            intended_blue[i] = (
                self.compute_target_pos(self.units_blue[i].pos, *delta, self.units_blue[i].speed)
                if typ == 'move' else self.units_blue[i].pos
            )
        
        # Resolve moves: conflict if >1 claimant to pos (cross-team)
        new_pos_to_slots = defaultdict(list)
        for slot, pos in intended_red.items():
            new_pos_to_slots[pos].append(('red', slot))
        for slot, pos in intended_blue.items():
            new_pos_to_slots[pos].append(('blue', slot))
        conflicted_pos = {pos for pos, claimants in new_pos_to_slots.items() if len(claimants) > 1}
        
        # Apply non-conflicted moves
        for i in range(5):
            target = intended_red[i]
            if target not in conflicted_pos:
                self.units_red[i].pos = target
        for i in range(5):
            target = intended_blue[i]
            if target not in conflicted_pos:
                self.units_blue[i].pos = target
        
        # Accumulate attack damages
        blue_unit_dmg: Dict[int, int] = defaultdict(int)
        blue_base_dmg = [0]
        self._resolve_attacks(self.units_red, actions_red, self.units_blue, (14, 14), self.bases_blue_alive,
                              blue_unit_dmg, blue_base_dmg)
        red_unit_dmg: Dict[int, int] = defaultdict(int)
        red_base_dmg = [0]
        self._resolve_attacks(self.units_blue, actions_blue, self.units_red, (0, 0), self.bases_red_alive,
                              red_unit_dmg, red_base_dmg)
        
        # Apply unit damages and count kills
        num_kills_red = 0  # blue units killed by red
        damage_dealt_red = sum(blue_unit_dmg.values()) + blue_base_dmg[0]
        for slot, dmg in blue_unit_dmg.items():
            if self.units_blue[slot].take_damage(dmg):
                num_kills_red += 1
        num_kills_blue = 0  # red units killed by blue
        damage_dealt_blue = sum(red_unit_dmg.values()) + red_base_dmg[0]
        for slot, dmg in red_unit_dmg.items():
            if self.units_red[slot].take_damage(dmg):
                num_kills_blue += 1
        
        # Apply base damages
        prev_red_alive = self.bases_red_alive
        prev_blue_alive = self.bases_blue_alive
        self.base_hp_red -= red_base_dmg[0]
        self.base_hp_blue -= blue_base_dmg[0]
        self.bases_red_alive = self.base_hp_red > 0
        self.bases_blue_alive = self.base_hp_blue > 0
        base_destroy_red = prev_red_alive and not self.bases_red_alive  # by blue
        base_destroy_blue = prev_blue_alive and not self.bases_blue_alive  # by red
        
        # Compute rewards
        rew_red = (
            15 * num_kills_red + 5 * damage_dealt_red
            - 15 * num_kills_blue - 5 * rev_red
            + (200 if base_destroy_blue else 0) - (200 if base_destroy_red else 0)
            + res_red_gain * 1
        )
        rew_blue = (
            15 * num_kills_blue + 5 * damage_dealt_blue
            - 15 * num_kills_red - 5 * rev_blue
            + (200 if base_destroy_red else 0) - (200 if base_destroy_blue else 0)
            + res_blue_gain * 1
        )
        
        # Update fog
        self.update_fog('red')
        self.update_fog('blue')
        
        # Check done and terminal rewards
        done = self.turn_count >= 500 or not (self.bases_red_alive and self.bases_blue_alive)
        if done:
            if self.bases_red_alive and not self.bases_blue_alive:
                rew_red += 100
                rew_blue -= 100
            elif self.bases_blue_alive and not self.bases_red_alive:
                rew_red -= 100
                rew_blue += 100
            # tie: 0
        
        return {'red': float(rew_red), 'blue': float(rew_blue)}, done
    def get_observation(self, agent: str) -> np.ndarray:
        '''Flatten 15x15x6 partial obs: terrain/2, own ID/4 hp/max, enemy ID/4 hp/max (if visible), own base + agent_id.'''
        fog = self.fog_red if agent == 'red' else self.fog_blue
        grid = np.zeros((15, 15, 6))
        grid[:, :, 0] = self.terrain.astype(float) / 2.0
        
        own_units = self.units_red if agent == 'red' else self.units_blue
        for u in own_units:
            r, c = u.pos
            if u.hp > 0 and fog[r, c]:
                grid[r, c, 1] = u.slot_id / 4.0
                grid[r, c, 2] = u.hp / u.max_hp
        
        enemy_units = self.units_blue if agent == 'red' else self.units_red
        for u in enemy_units:
            r, c = u.pos
            if u.hp > 0 and fog[r, c]:
                grid[r, c, 3] = u.slot_id / 4.0
                grid[r, c, 4] = u.hp / u.max_hp
        
        base_pos = (0, 0) if agent == 'red' else (14, 14)
        own_base_alive = self.bases_red_alive if agent == 'red' else self.bases_blue_alive
        br, bc = base_pos
        if own_base_alive and fog[br, bc]:
            grid[br, bc, 5] = 1.0
        
        agent_id = 0.0 if agent == 'red' else 1.0
        return np.append(grid.flatten(), agent_id).astype(np.float32)
