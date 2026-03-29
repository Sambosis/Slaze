import numpy as np
import random


class Ship:
    """
    Represents a single ship with its positions and sink status on a board.
    Attributes:
        length (int): Length of the ship.
        positions (list[tuple[int,int]]): List of (row, col) positions occupied by the ship.
        orientation (str): 'H' for horizontal or 'V' for vertical.
        sunk (bool): Whether the ship has been fully sunk.
    """
    def __init__(self, length: int, positions: list[tuple[int, int]], orientation: str):
        self.length = length
        self.positions = positions
        self.orientation = orientation
        self.sunk = False


class BattleshipBoard:
    """
    Manages a player's board state, including ship placements, hits/misses, and sink tracking.
    Attributes:
        grid (np.ndarray shape=(10,10) dtype=int): 0=empty water, 1=undamaged ship, 2=damaged ship.
        ships (list[Ship]): List of placed ships.
        shots_received (set[tuple[int,int]]): Set of positions shot at.
    """
    def __init__(self, size: int = 10):
        self.size = size
        self.grid = np.zeros((size, size), dtype=np.int32)
        self.ships = []
        self.shots_received = set()

    def place_ship(self, ship: Ship) -> None:
        """Place a ship on the board by marking its positions as undamaged (1)."""
        for r, c in ship.positions:
            self.grid[r, c] = 1
        self.ships.append(ship)

    def shoot(self, row: int, col: int) -> tuple[bool, bool, int]:
        """
        Process a shot at (row, col).
        Returns: (reshot: bool, hit: bool, sunk_length: int)
        """
        if row < 0 or row >= self.size or col < 0 or col >= self.size:
            return True, False, 0  # Invalid shot treated as reshot
        pos = (row, col)
        if pos in self.shots_received:
            return True, False, 0
        self.shots_received.add(pos)
        was_hit = self.grid[row, col] == 1
        hit = False
        sunk_length = 0
        if was_hit:
            self.grid[row, col] = 2
            hit = True
            # Check if this hit sunk a ship
            for ship in self.ships:
                if (not ship.sunk and
                    pos in ship.positions and
                    all(self.grid[pr, pc] == 2 for pr, pc in ship.positions)):
                    ship.sunk = True
                    sunk_length = ship.length
                    break
        return False, hit, sunk_length


class BattleshipGame:
    """
    Core game engine handling two boards, turns, shots, rewards, and states for both players.
    """
    def __init__(self, board_size: int = 10, ship_lengths: list[int] = [2, 3, 3, 4, 5]):
        self.board_size = board_size
        self.ship_lengths = ship_lengths
        self.board_a = None
        self.board_b = None
        self.current_player = 'A'
        self.turn_count = 0

    def reset(self) -> None:
        """Reset the game: create new boards and randomly place ships until successful."""
        # Place ships on board A
        while True:
            self.board_a = BattleshipBoard(self.board_size)
            if random_place_ships(self.board_a, self.ship_lengths):
                break
        # Place ships on board B
        while True:
            self.board_b = BattleshipBoard(self.board_size)
            if random_place_ships(self.board_b, self.ship_lengths):
                break
        self.current_player = 'A'
        self.turn_count = 0

    def shoot(self, player: str, row: int, col: int) -> tuple[float, bool, int]:
        """
        Execute a shot for the given player.
        Returns: (reward: float, hit: bool, sunk_length: int)
        """
        if self.current_player != player:
            return -1.0, False, 0  # Wrong turn
        opp_board = self.board_b if player == 'A' else self.board_a
        reshot, hit, sunk_length = opp_board.shoot(row, col)
        # Compute reward
        reward = -0.1  # Step penalty
        if reshot:
            reward = -1.0
        elif not hit:
            reward = -1.0
        else:
            reward += 5.0  # Hit reward
        if sunk_length > 0:
            reward += 20.0 * sunk_length  # Sunk ship reward (scaled by size)
        # Check terminal
        done, winner = self.is_terminal()
        if done:
            if winner == player:
                reward += 200.0  # Win bonus
            elif winner is not None and winner != 'draw':
                reward -= 200.0  # Loss penalty
        # Switch turn
        self.current_player = 'B' if player == 'A' else 'A'
        self.turn_count += 1
        return reward, hit, sunk_length

    def is_terminal(self) -> tuple[bool, str]:
        """
        Check if game is over.
        Returns: (done: bool, winner: str or None)  # 'A', 'B', 'draw', or None
        """
        if self.board_a is None or self.board_b is None:
            return False, None
        all_sunk_a = all(s.sunk for s in self.board_a.ships)
        all_sunk_b = all(s.sunk for s in self.board_b.ships)
        if all_sunk_a and all_sunk_b:
            return True, 'draw'
        elif all_sunk_a:
            return True, 'B'
        elif all_sunk_b:
            return True, 'A'
        return False, None

    def get_state(self, player: str) -> np.ndarray:
        """
        Get the state for a player: flattened [own_board (10x10), opp_view (10x10)] -> 200-dim float32.
        Own: 0=water, 1=undamaged ship, 2=damaged.
        Opp: 0=unprobed, 1=miss, 2=hit.
        """
        if player == 'A':
            own_grid = self.board_a.grid
            opp_board = self.board_b
        else:
            own_grid = self.board_b.grid
            opp_board = self.board_a
        opp_view = np.zeros((self.board_size, self.board_size), dtype=np.int32)
        for r, c in opp_board.shots_received:
            if 0 <= r < self.board_size and 0 <= c < self.board_size:
                opp_view[r, c] = 2 if opp_board.grid[r, c] == 2 else 1
        state = np.hstack((own_grid.flatten(), opp_view.flatten())).astype(np.float32)
        return state


def random_place_ships(board: BattleshipBoard, ship_lengths: list[int], max_attempts: int = 100) -> bool:
    """
    Randomly places all ships on a board without overlap using backtracking random trials.
    Shuffles ship order, tries up to 50 positions per ship.
    Returns success flag.
    """
    size = board.size
    lengths = ship_lengths[:]
    random.shuffle(lengths)
    attempts = 0
    for length in lengths:
        placed = False
        for _ in range(50):
            attempts += 1
            if attempts > max_attempts:
                return False
            orient = random.choice(['H', 'V'])
            if orient == 'H':
                r = random.randint(0, size - 1)
                c_start = random.randint(0, size - length)
                positions = [(r, c_start + j) for j in range(length)]
            else:
                r_start = random.randint(0, size - length)
                c = random.randint(0, size - 1)
                positions = [(r_start + j, c) for j in range(length)]
            # Verify bounds and no overlap
            if all(0 <= rr < size and 0 <= cc < size and board.grid[rr, cc] == 0
                   for rr, cc in positions):
                ship = Ship(length, positions, orient)
                board.place_ship(ship)
                placed = True
                break
        if not placed:
            return False
    return True