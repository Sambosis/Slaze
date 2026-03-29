from typing import Optional
import heapq
from renderer import Renderer
from ai import AI, pathfind_a_star
import pygame
import random
import math

TERRAIN_INFO = {
    'grass': {'passable': True, 'defense_bonus': 0.0},
    'forest': {'passable': True, 'defense_bonus': 0.5},
    'mountain': {'passable': False, 'defense_bonus': 1.0},
    'river': {'passable': True, 'defense_bonus': 0.25},
}

UNIT_STATS = {
    'infantry': {'hp': 10, 'move': 2, 'atk': 5, 'range': 1, 'cost': 1},
    'tank': {'hp': 20, 'move': 3, 'atk': 10, 'range': 1, 'cost': 3},
    'artillery': {'hp': 15, 'move': 1, 'atk': 15, 'range': 3, 'cost': 2},
}

class Tile:
    """Defines a single map tile's properties for movement, combat, and capture."""
    def __init__(self, terrain: str, owner: str = 'neutral', hq: bool = False):
        self.terrain = terrain
        self.owner = owner
        self.hq = hq
        self.update_props()
        self.hp = 50 if hq else 9999
        self.max_hp = 50 if hq else 9999

    def update_props(self):
        """Update passable and defense_bonus based on current terrain."""
        props = TERRAIN_INFO[self.terrain]
        self.passable = props['passable']
        self.defense_bonus = props['defense_bonus']

class Board:
    """Represents the game map as a 2D grid of tiles with terrain types and ownership."""
    def __init__(self, width: int = 15, height: int = 10, tile_size: int = 48):
        self.width = width
        self.height = height
        self.tile_size = tile_size
        self.grid = []

    def get_tile(self, x: int, y: int):
        """Get tile at position (x, y), or None if out of bounds."""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[x][y]
        return None

    def generate_terrain(self):
        """Procedurally generates terrain ensuring connectivity between HQs."""
        random.seed(42)
        terrain_probs = ['grass'] * 50 + ['forest'] * 20 + ['mountain'] * 15 + ['river'] * 15
        connected = False
        for attempt in range(20):
            # Generate initial random terrain
            terrains = [[random.choice(terrain_probs) for _ in range(self.height)] for _ in range(self.width)]
            # Smooth terrain for clustering
            for _ in range(5):
                new_terrains = [[terrains[x][y] for y in range(self.height)] for x in range(self.width)]
                for x in range(self.width):
                    for y in range(self.height):
                        neighbors = []
                        for dx in range(-1, 2):
                            for dy in range(-1, 2):
                                nx, ny = x + dx, y + dy
                                if 0 <= nx < self.width and 0 <= ny < self.height:
                                    neighbors.append(terrains[nx][ny])
                        if neighbors:
                            most_common = max(set(neighbors), key=neighbors.count)
                            if neighbors.count(most_common) >= 5:
                                new_terrains[x][y] = most_common
                terrains = new_terrains
            # Check connectivity via passable tiles
            passable_grid = [[TERRAIN_INFO[terrains[x][y]]['passable'] for y in range(self.height)] for x in range(self.width)]
            if self._is_connected(passable_grid):
                connected = True
                break
        if not connected:
            print("Warning: Could not generate fully connected map after 20 attempts.")
        # Create Tile grid
        self.grid = [[Tile(terrains[x][y]) for y in range(self.height)] for x in range(self.width)]
        # Override HQs
        red_hq = self.grid[0][0]
        red_hq.terrain = 'grass'
        red_hq.owner = 'red'
        red_hq.hq = True
        red_hq.update_props()
        red_hq.hp = 50
        red_hq.max_hp = 50
        blue_hq = self.grid[14][9]
        blue_hq.terrain = 'grass'
        blue_hq.owner = 'blue'
        blue_hq.hq = True
        blue_hq.update_props()
        blue_hq.hp = 50
        blue_hq.max_hp = 50

    def _is_connected(self, passable_grid):
        """BFS/DFS check if blue HQ reachable from red HQ via passable tiles."""
        visited = set()
        stack = [(0, 0)]
        while stack:
            cx, cy = stack.pop()
            if (cx, cy) in visited:
                continue
            visited.add((cx, cy))
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.width and 0 <= ny < self.height and passable_grid[nx][ny] and (nx, ny) not in visited:
                    stack.append((nx, ny))
        return (14, 9) in visited

class Unit:
    """Models a combat unit with stats, position, and animation state for movement/attacks."""
    def __init__(self, unit_type: str, player: str, pos: tuple[int, int]):
        self.unit_type = unit_type
        self.player = player
        self.pos = pos
        stats = UNIT_STATS[unit_type]
        self.hp = stats['hp']
        self.max_hp = stats['hp']
        self.move_range = stats['move']
        self.atk = stats['atk']
        self.atk_range = stats['range']
        self.cost = stats['cost']
        self.target_pos = pos
        self.anim_progress = 0.0

class Game:
    """Manages overall game state, turn alternation, win conditions, and main loop."""
    def __init__(self, screen: pygame.Surface, clock: pygame.time.Clock):
        self.screen = screen
        self.clock = clock
        self.board = Board()
        self.board.generate_terrain()
        self.units = []
        self.current_player = 'red'
        self.turn_count = 0
        self.funds_red = 5
        self.funds_blue = 5
        self.ai = AI()
        self.renderer = None

    def spawn_unit(self, unit_type: str, pos: tuple[int, int]) -> Unit | None:
        """Spawn a unit if affordable and under limit."""
        player = self.current_player
        player_units = [u for u in self.units if u.player == player]
        if len(player_units) >= 10:
            return None
        cost = UNIT_STATS[unit_type]['cost']
        funds = self.funds_red if player == 'red' else self.funds_blue
        if funds < cost:
            return None
        unit = Unit(unit_type, player, pos)
        self.units.append(unit)
        if player == 'red':
            self.funds_red -= cost
        else:
            self.funds_blue -= cost
        print(f"{player.capitalize()} spawned {unit_type} at {pos} (funds left: {funds - cost})")
        return unit

    def end_turn_on_tile(self, unit: Unit):
        """Capture enemy/neutral non-HQ tile by ending turn on it."""
        tile = self.board.get_tile(*unit.pos)
        if tile is None or tile.hq or tile.owner == unit.player:
            return
        tile.owner = unit.player
        print(f"{unit.player.capitalize()} captured {tile.terrain} tile at {unit.pos}")

    def check_win(self) -> str | None:
        """Check win condition: opponent's HQ HP <= 0."""
        red_hq = self.board.get_tile(0, 0)
        blue_hq = self.board.get_tile(14, 9)
        if red_hq and red_hq.hp <= 0:
            return 'blue'
        if blue_hq and blue_hq.hp <= 0:
            return 'red'
        if self.turn_count > 500:
            return 'tie'
        return None

    def update(self, dt: float):
        """Update unit animations (tween progress)."""
        for unit in self.units[:]:  # copy to avoid mod during iter
            if unit.anim_progress < 1.0:
                unit.anim_progress = min(1.0, unit.anim_progress + dt / 0.3)
                if unit.anim_progress >= 1.0:
                    unit.pos = unit.target_pos
                    unit.anim_progress = 0.0

    def collect_funds(self):
        """Add +1 funds per owned tile (excluding neutral)."""
        red_count = sum(1 for row in self.board.grid for tile in row if tile.owner == 'red')
        blue_count = sum(1 for row in self.board.grid for tile in row if tile.owner == 'blue')
        self.funds_red += red_count
        self.funds_blue += blue_count
        print(f"Funds - Red: {self.funds_red}, Blue: {self.funds_blue} (+{red_count}, +{blue_count})")

    def remove_dead_units(self):
        """Remove units with HP <= 0."""
        initial_count = len(self.units)
        self.units = [u for u in self.units if u.hp > 0]
        if len(self.units) < initial_count:
            print(f"Removed {initial_count - len(self.units)} dead units")

    def perform_ai_turn(self):
        """Perform AI turn: collect funds, spawn, actions (placeholder), capture, end turn."""
        self.collect_funds()
        print(f"--- {self.current_player.capitalize()}'s Turn {self.turn_count} ---")
        hq_pos = (0, 0) if self.current_player == 'red' else (14, 9)
        # Spawn 1-3 units if under limit
        num_spawns = random.randint(1, min(3, 10 - len([u for u in self.units if u.player == self.current_player])))
        for _ in range(num_spawns):
            spawn_type = self.ai.get_spawn_type(self)
            if spawn_type and self.spawn_unit(spawn_type, hq_pos):
                pass
        # AI actions up to 8
        action_count = 0
        while action_count < 8:
            action = self.ai.get_best_action(self, self.units)
            if not action:
                break
            self._execute_action(action)
            action_count += 1
        # Capture
        own_units = [u for u in self.units if u.player == self.current_player]
        for unit in own_units:
            self.end_turn_on_tile(unit)
        self.remove_dead_units()
        self.turn_count += 1
        self.current_player = 'blue' if self.current_player == 'red' else 'red'
    def _execute_action(self, action):
        unit_idx = action['unit_id']
        unit = self.units[unit_idx]
        act = action['action']
        target = action['target']
        if act == 'move':
            path = pathfind_a_star(self.board, unit.pos, target)
            if path and len(path) - 1 <= unit.move_range:
                unit.target_pos = target
                unit.anim_progress = 0.0
        elif act == 'attack':
            target_tile = self.board.get_tile(*target)
            if target_tile:
                defender = next((u for u in self.units if u.pos == target and u.player != self.current_player), None)
                if defender:
                    bonus = target_tile.defense_bonus
                    _, new_hp = resolve_combat(unit, defender, bonus)
                    defender.hp = new_hp
                    if defender.hp <= 0:
                        print(f"{unit.player.capitalize()} killed enemy {defender.unit_type}!")
                    if self.renderer:
                        self.renderer.add_explosion(target)
                        self.renderer.trigger_shake(0.3)
                elif target_tile.hq and target_tile.owner != self.current_player:
                    bonus = target_tile.defense_bonus
                    damage = max(1, unit.atk - int(bonus + 0.5))
                    target_tile.hp = max(0, target_tile.hp - damage)
                    print(f"{unit.player.capitalize()} attacked enemy HQ for {damage} damage. Remaining HP: {target_tile.hp}")
                    if self.renderer:
                        self.renderer.add_explosion(target)
                        self.renderer.trigger_shake(0.3)

def resolve_combat(attacker: Unit, defender: Unit, tile_bonus: float) -> tuple[int, int]:
    """
    Resolve unit-on-unit combat.
    Returns (attacker_new_hp, defender_new_hp). Attacker takes no damage.
    """
    damage = attacker.atk * (1 - tile_bonus)
    new_def_hp = max(0, defender.hp - int(damage))
    return attacker.hp, new_def_hp