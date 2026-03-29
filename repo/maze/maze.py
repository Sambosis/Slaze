import random
from typing import List, Tuple

class Maze:
    """Represents a maze with its grid, dimensions, and start/end points."""

    def __init__(self, grid: List[List[int]], width: int, height: int,
                 start: Tuple[int, int], end: Tuple[int, int]):
        """Initialize the maze with grid, dimensions, and start/end points."""
        self.grid = grid
        self.width = width
        self.height = height
        self.start = start
        self.end = end

def generate_maze(width: int, height: int) -> Maze:
    """Generate a maze using randomized depth-first search algorithm.

    Args:
        width: Number of columns in the maze (must be at least 3)
        height: Number of rows in the maze (must be at least 3)

    Returns:
        Maze object with generated grid and start/end points
    """
    if width < 3 or height < 3:
        raise ValueError("Maze dimensions must be at least 3x3")

    # Ensure odd dimensions for proper maze generation
    if width % 2 == 0:
        width += 1
    if height % 2 == 0:
        height += 1

    # Initialize grid with walls (1)
    grid = [[1 for _ in range(width)] for _ in range(height)]

    # Start from a random odd position to ensure it's a path cell
    start_x = random.randrange(1, width, 2)
    start_y = random.randrange(1, height, 2)
    grid[start_y][start_x] = 0

    # Use stack for DFS
    stack = [(start_x, start_y)]

    # Possible movement directions (dx, dy)
    directions = [(0, -2), (2, 0), (0, 2), (-2, 0)]

    while stack:
        x, y = stack[-1]
        neighbors = []

        # Find all unvisited neighbors (two steps away)
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 < nx < width - 1 and 0 < ny < height - 1 and grid[ny][nx] == 1:
                neighbors.append((nx, ny))

        if neighbors:
            # Choose random neighbor
            nx, ny = random.choice(neighbors)
            # Carve path to neighbor
            grid[ny][nx] = 0
            # Carve the wall between current cell and neighbor
            grid[(y + ny) // 2][(x + nx) // 2] = 0
            stack.append((nx, ny))
        else:
            stack.pop()

    # Find start position (top-left path cell)
    start = None
    for y in range(height):
        for x in range(width):
            if grid[y][x] == 0:
                start = (x, y)
                break
        if start is not None:
            break

    # Find end position (bottom-right path cell)
    end = None
    for y in range(height - 1, -1, -1):
        for x in range(width - 1, -1, -1):
            if grid[y][x] == 0:
                end = (x, y)
                break
        if end is not None:
            break

    return Maze(grid, width, height, start, end)