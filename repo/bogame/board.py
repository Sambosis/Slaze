import pygame
import random


class Ship:
    """
    Represents a single ship with position, damage tracking, and sink status for Battleship logic.
    Attributes:
        length (int): Length of the ship.
        pos (tuple[int, int]): Starting position (x, y).
        orientation (str): 'h' for horizontal, 'v' for vertical.
        hits (int): Number of hits taken.
        sunk (bool): Whether the ship is sunk.
    Methods:
        __init__(length: int, pos: tuple[int,int], orient: str)
        hit() -> bool: Increments hits and returns True if sunk.
    """

    def __init__(self, length: int, pos: tuple[int, int], orient: str):
        self.length = length
        self.pos = pos
        self.orientation = orient
        self.hits = 0
        self.sunk = False

    def hit(self) -> bool:
        """Increments hits and sets sunk if fully hit. Returns if sunk."""
        self.hits += 1
        if self.hits == self.length:
            self.sunk = True
        return self.sunk


class Board:
    """
    Manages 10x10 grid states, random ship placement, hit/miss probing, sink detection, rendering overlays/grid.
    Attributes:
        size (int): Grid size (default 10).
        cell_size (int): Pixel size per cell (default 50).
        grid (list[list[str]]): 2D list of cell states: 'empty', 'ship', 'hit', 'miss'.
        ships (list[Ship]): List of placed ships.
        sunk_ships (int): Number of sunk ships.
        hit_img (pygame.Surface): Cached hit overlay image.
        miss_img (pygame.Surface): Cached miss overlay image.
    Methods:
        place_ships() -> None: Randomly places 4 ships without overlaps or adjacency.
        probe(x: int, y: int) -> tuple[bool, bool]: Probes cell, returns (is_hit, is_sink).
        is_valid_pos(x: int, y: int) -> bool: Checks if position is on grid.
        draw(screen: pygame.Surface, offset=(150, 50)) -> None: Draws grid lines and overlays.
    """

    def __init__(self, size: int = 10, cell_size: int = 50):
        self.size = size
        self.cell_size = cell_size
        self.grid = [['empty' for _ in range(size)] for _ in range(size)]
        self.ships = []
        self.sunk_ships = 0
        try:
            self.hit_img = pygame.image.load('assets/hit.png').convert_alpha()
            self.miss_img = pygame.image.load('assets/miss.png').convert_alpha()
        except pygame.error as e:
            print(f"Warning: Could not load overlay images: {e}")
            self.hit_img = None
            self.miss_img = None

    def is_valid_pos(self, x: int, y: int) -> bool:
        """Checks if (x, y) is within grid bounds."""
        return 0 <= x < self.size and 0 <= y < self.size
    def place_ships(self) -> None:
        """
    Places 4 ships randomly without overlaps, preferring no orthogonal adjacency.
    Reliable on 10x10 grid with high attempts + fallback ignoring adjacency.
    """
        # Reset
        self.grid = [['empty' for _ in range(self.size)] for _ in range(self.size)]
        self.ships = []
        self.sunk_ships = 0
        
        ship_lengths = [2, 3, 3, 4]
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # orthogonal only
        
        for length in ship_lengths:
            placed = False
            for attempt in range(1000):  # High attempts
                orientation = random.choice(['h', 'v'])
                if orientation == 'h':
                    sx = random.randint(0, self.size - length)
                    sy = random.randint(0, self.size - 1)
                    dx, dy = 1, 0
                else:
                    sx = random.randint(0, self.size - 1)
                    sy = random.randint(0, self.size - length)
                    dx, dy = 0, 1
                
                # Check overlap
                positions = []
                valid = True
                for i in range(length):
                    px = sx + i * dx
                    py = sy + i * dy
                    if not self.is_valid_pos(px, py) or self.grid[py][px] != 'empty':
                        valid = False
                        break
                    positions.append((px, py))
                if not valid:
                    continue
                
                # Prefer no adjacency
                adj_valid = True
                for px, py in positions:
                    for ddx, ddy in directions:
                        ax = px + ddx
                        ay = py + ddy
                        if self.is_valid_pos(ax, ay) and self.grid[ay][ax] == 'ship':
                            adj_valid = False
                            break
                    if not adj_valid:
                        break
                if adj_valid:
                    # Place
                    for px, py in positions:
                        self.grid[py][px] = 'ship'
                    self.ships.append(Ship(length, (sx, sy), orientation))
                    placed = True
                    break
            
            if not placed:
                # Fallback: place without adj check, find contiguous empty cells
                print(f"Info: Fallback placement for ship length {length}")
                orientation = random.choice(['h', 'v'])
                if orientation == 'h':
                    for sy in range(self.size):
                        for sx in range(self.size - length + 1):
                            valid = all(self.grid[sy][sx + i] == 'empty' for i in range(length))
                            if valid:
                                for i in range(length):
                                    self.grid[sy][sx + i] = 'ship'
                                self.ships.append(Ship(length, (sx, sy), 'h'))
                                placed = True
                                break
                        if placed:
                            break
                else:  # 'v'
                    for sx in range(self.size):
                        for sy in range(self.size - length + 1):
                            valid = all(self.grid[sy + i][sx] == 'empty' for i in range(length))
                            if valid:
                                for i in range(length):
                                    self.grid[sy + i][sx] = 'ship'
                                self.ships.append(Ship(length, (sx, sy), 'v'))
                                placed = True
                                break
                        if placed:
                            break
                
                if not placed:
                    raise ValueError(f"Failed to place ship of length {length} even with fallback")

    def probe(self, x: int, y: int) -> tuple[bool, bool]:
        """
        Probes cell (x,y):
        - If 'ship': hits the ship, marks 'hit', returns (True, sunk_this_hit)
        - If 'empty': marks 'miss', returns (False, False)
        - Else: no change, (False, False)
        """
        if not self.is_valid_pos(x, y):
            return False, False

        state = self.grid[y][x]
        if state == 'ship':
            for ship in self.ships:
                covers = False
                sx, sy = ship.pos
                if ship.orientation == 'h':
                    if sy == y and sx <= x < sx + ship.length:
                        covers = True
                elif ship.orientation == 'v':
                    if sx == x and sy <= y < sy + ship.length:
                        covers = True
                if covers:
                    was_sunk = ship.hit()
                    self.grid[y][x] = 'hit'
                    if was_sunk:
                        self.sunk_ships += 1
                    return True, was_sunk
        elif state == 'empty':
            self.grid[y][x] = 'miss'
            return False, False
        return False, False
    def draw(self, screen: pygame.Surface, offset=(150, 50)) -> None:
        """Draws faint grid lines and hit/miss overlays at given offset."""
        ox, oy = offset
        # Faint white grid lines
        line_color = (200, 200, 220)
        line_width = 1

        # Vertical lines
        for i in range(self.size + 1):
            x = ox + i * self.cell_size
            pygame.draw.line(screen, line_color, (x, oy), (x, oy + self.size * self.cell_size), line_width)
        # Horizontal lines
        for i in range(self.size + 1):
            y = oy + i * self.cell_size
            pygame.draw.line(screen, line_color, (ox, y), (ox + self.size * self.cell_size, y), line_width)

        # Overlays for hit/miss
        if self.hit_img and self.miss_img:
            for y in range(self.size):
                for x in range(self.size):
                    state = self.grid[y][x]
                    if state == 'hit':
                        px = ox + x * self.cell_size
                        py = oy + y * self.cell_size
                        screen.blit(self.hit_img, (px, py))
                    elif state == 'miss':
                        px = ox + x * self.cell_size
                        py = oy + y * self.cell_size
                        screen.blit(self.miss_img, (px, py))