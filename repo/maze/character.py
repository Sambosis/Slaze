import heapq
from typing import List, Tuple, Dict, Set
from maze import Maze

class Character:
    """
    Represents the character navigating the maze.

    Attributes:
        position (Tuple[int, int]): Current (row, col) position in the maze grid.
        path (List[Tuple[int, int]]): List of positions representing the path to follow.
        speed (float): Movement speed in cells per second.
        _path_index (int): Current index in the path.
        _time_since_last_move (float): Accumulated time since last movement.
    """

    def __init__(self, start_position: Tuple[int, int], speed: float = 2.0):
        """
        Initialize the character with a starting position and speed.

        Args:
            start_position (Tuple[int, int]): Starting (row, col) position.
            speed (float): Movement speed in cells per second. Default is 2.0.
        """
        self.position = start_position
        self.path: List[Tuple[int, int]] = []
        self.speed = speed
        self._path_index = 0
        self._time_since_last_move = 0.0

    def set_path(self, path: List[Tuple[int, int]]) -> None:
        """
        Set a new path for the character to follow.

        Args:
            path (List[Tuple[int, int]]): New path to follow.
        """
        self.path = path
        self._path_index = 0
        if path:
            self.position = path[0]

    def update(self, dt: float) -> bool:
        """
        Update the character's position based on elapsed time and path.

        Args:
            dt (float): Time delta in seconds since last update.

        Returns:
            bool: True if the character moved, False otherwise.
        """
        if not self.path or self._path_index >= len(self.path):
            return False

        self._time_since_last_move += dt
        time_per_cell = 1.0 / self.speed
        moved = False

        while self._time_since_last_move >= time_per_cell:
            self._time_since_last_move -= time_per_cell
            self._path_index += 1

            if self._path_index < len(self.path):
                self.position = self.path[self._path_index]
                moved = True
            else:
                break

        return moved

    def has_reached_destination(self) -> bool:
        """
        Check if the character has reached the end of its path.

        Returns:
            bool: True if at destination, False otherwise.
        """
        return self._path_index >= len(self.path)

def _heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    """
    Calculate Manhattan distance between two points.

    Args:
        a (Tuple[int, int]): First point (row, col).
        b (Tuple[int, int]): Second point (row, col).

    Returns:
        int: Manhattan distance.
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def find_path(maze: Maze, start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    Find the shortest path from start to end using A* algorithm.

    Args:
        maze (Maze): Maze instance containing the grid.
        start (Tuple[int, int]): Starting position (row, col).
        end (Tuple[int, int]): Ending position (row, col).

    Returns:
        List[Tuple[int, int]]: Path as list of positions, or empty list if no path exists.
    """
    if not (0 <= start[0] < maze.height and 0 <= start[1] < maze.width) or \
       not (0 <= end[0] < maze.height and 0 <= end[1] < maze.width):
        return []

    if maze.grid[start[0]][start[1]] != 0 or maze.grid[end[0]][end[1]] != 0:
        return []

    open_set: List[Tuple[float, Tuple[int, int]]] = []
    heapq.heappush(open_set, (0, start))

    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    g_score: Dict[Tuple[int, int], float] = {start: 0}
    f_score: Dict[Tuple[int, int], float] = {start: _heuristic(start, end)}
    closed_set: Set[Tuple[int, int]] = set()

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while open_set:
        current_f, current = heapq.heappop(open_set)

        if current == end:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        if current in closed_set:
            continue

        closed_set.add(current)

        for dr, dc in directions:
            neighbor = (current[0] + dr, current[1] + dc)

            if not (0 <= neighbor[0] < maze.height and 0 <= neighbor[1] < maze.width):
                continue

            if maze.grid[neighbor[0]][neighbor[1]] != 0:
                continue

            tentative_g = g_score[current] + 1

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + _heuristic(neighbor, end)

                if neighbor not in closed_set:
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return []