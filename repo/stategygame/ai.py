import heapq
import math
from typing import Dict, List, Tuple, Optional

from game import Game, Board, Tile, Unit


def manhattan(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
    """Calculate Manhattan distance between two positions."""
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def pathfind_a_star(board: Board, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    Computes shortest passable path between tiles using A* algorithm.
    Ignores units, considers terrain costs: grass=1.0, forest=0.8, river=1.5, mountain=inf.
    Heuristic: Manhattan distance.
    """
    if (start[0] < 0 or start[0] >= board.width or
        start[1] < 0 or start[1] >= board.height or
        goal[0] < 0 or goal[0] >= board.width or
        goal[1] < 0 or goal[1] >= board.height):
        return []

    start_tile = board.get_tile(*start)
    goal_tile = board.get_tile(*goal)
    if not start_tile.passable or not goal_tile.passable:
        return []

    terrain_costs = {
        'grass': 1.0,
        'forest': 0.8,
        'river': 1.5,
        'mountain': float('inf')
    }

    def get_cost(tile: Tile) -> float:
        return terrain_costs.get(tile.terrain, 1.0)

    def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return manhattan(a, b) * 0.8  # Conservative heuristic (min cost)

    open_set: List[Tuple[float, int, Tuple[int, int], float]] = []
    counter = 0
    heapq.heappush(open_set, (heuristic(start, goal), counter, start, 0.0))  # (f, counter, pos, g)

    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    g_score: Dict[Tuple[int, int], float] = {start: 0.0}
    f_score: Dict[Tuple[int, int], float] = {start: heuristic(start, goal)}

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while open_set:
        _, _, current, current_g = heapq.heappop(open_set)

        if current == goal:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path

        for dx, dy in directions:
            nx = current[0] + dx
            ny = current[1] + dy
            if nx < 0 or nx >= board.width or ny < 0 or ny >= board.height:
                continue
            neighbor = (nx, ny)
            neighbor_tile = board.get_tile(nx, ny)
            if not neighbor_tile.passable:
                continue

            move_cost = get_cost(neighbor_tile)
            tent_g = current_g + move_cost

            if neighbor in g_score and tent_g >= g_score[neighbor]:
                continue

            came_from[neighbor] = current
            g_score[neighbor] = tent_g
            f_score_val = tent_g + heuristic(neighbor, goal)
            f_score[neighbor] = f_score_val
            counter += 1
            heapq.heappush(open_set, (f_score_val, counter, neighbor, tent_g))

    return []


class AI:
    """
    Implements rule-based decision-making for unit actions and spawning.
    """

    UNIT_STATS = {
        'infantry': (10, 2, 5, 1, 1),   # max_hp, move_range, atk, atk_range, cost
        'artillery': (15, 1, 15, 3, 2),
        'tank': (20, 3, 10, 1, 3),
    }

    def get_best_action(self, game: Game, units: List[Unit]) -> Optional[Dict]:
        """
        Returns the best action for the current player: {'unit_id': int, 'action': str, 'target': tuple[int,int]}
        Priorities: 1. Attack weakest/closest enemy or HQ in range.
                    2. Move to capture neutral/enemy tiles or advance toward enemy units/HQ.
        Returns None if no viable action.
        unit_id is index in the provided units list.
        """
        current_player = game.current_player
        opponent = 'blue' if current_player == 'red' else 'red'
        board = game.board
        opp_hq_pos = (14, 9) if current_player == 'red' else (0, 0)

        own_units = [u for u in units if u.player == current_player]
        enemy_units = [u for u in units if u.player == opponent]

        if not own_units:
            return None

        # Priority 1: Best attack
        best_action = None
        best_score = -1.0

        for i, unit in enumerate(units):
            if unit.player != current_player:
                continue

            # Attack enemy units
            for enemy in enemy_units:
                dist = manhattan(unit.pos, enemy.pos)
                if dist > unit.atk_range or dist == 0:
                    continue
                enemy_tile = board.get_tile(*enemy.pos)
                tile_bonus = enemy_tile.defense_bonus
                damage = max(1, unit.atk - int(tile_bonus + 0.5))
                hp_ratio = enemy.hp / max(1, enemy.max_hp)
                score = damage * (1.0 - hp_ratio) / (dist + 1)
                if score > best_score:
                    best_score = score
                    best_action = {'unit_id': i, 'action': 'attack', 'target': enemy.pos}

            # Attack opponent HQ
            hq_dist = manhattan(unit.pos, opp_hq_pos)
            if hq_dist <= unit.atk_range and hq_dist > 0:
                hq_tile = board.get_tile(*opp_hq_pos)
                hq_bonus = hq_tile.defense_bonus
                hq_damage = max(1, unit.atk - int(hq_bonus + 0.5))
                score = hq_damage * 1.5 / (hq_dist + 1)  # HQ priority boost
                if score > best_score:
                    best_score = score
                    best_action = {'unit_id': i, 'action': 'attack', 'target': opp_hq_pos}

        if best_action:
            return best_action

        # Priority 2: Best move
        best_score = -1.0
        max_range = max(u.move_range for u in own_units) if own_units else 0

        for i, unit in enumerate(units):
            if unit.player != current_player:
                continue

            candidates: List[Tuple[int, int]] = [opp_hq_pos]
            candidates.extend([e.pos for e in enemy_units])

            # Add nearby neutral/enemy passable tiles
            for x in range(board.width):
                for y in range(board.height):
                    gpos = (x, y)
                    if manhattan(unit.pos, gpos) > unit.move_range + 1:
                        continue
                    tile = board.get_tile(x, y)
                    if tile.passable and tile.owner in ('neutral', opponent):
                        candidates.append(gpos)

            for goal in candidates:
                if goal == unit.pos:
                    continue
                path = pathfind_a_star(board, unit.pos, goal)
                if not path or len(path) - 1 > unit.move_range:
                    continue
                move_dist = len(path) - 1
                goal_tile = board.get_tile(*goal)
                capture_val = 10 if goal_tile.owner == opponent else 5 if goal_tile.owner == 'neutral' else 0
                threat_dist = manhattan(goal, opp_hq_pos)
                score = (capture_val + 10.0 / (threat_dist + 1)) / (move_dist + 1)
                if score > best_score:
                    best_score = score
                    best_action = {'unit_id': i, 'action': 'move', 'target': goal}

        return best_action
    def get_spawn_type(self, game: Game) -> Optional[str]:
        player = game.current_player
        funds = game.funds_red if player == 'red' else game.funds_blue
        if funds < 1:
            return None
        affordable = [utype for utype, stats in self.UNIT_STATS.items() if funds >= stats[4]]
        if not affordable:
            return None
        affordable.sort(key=lambda u: self.UNIT_STATS[u][4])
        if funds < 3:
            choice = 'infantry' if 'infantry' in affordable else affordable[0]
        else:
            idx = game.turn_count % len(affordable)
            choice = affordable[idx]
        return choice