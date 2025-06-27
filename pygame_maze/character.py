
import pygame
import config
from collections import deque

class Character:
    def __init__(self, x, y, maze):
        self.x = x
        self.y = y
        self.maze = maze
        self.path = self._find_path()
        self.path_index = 0

    def _find_path(self):
        start = self.maze.start_node
        end = self.maze.end_node
        queue = deque([(start, [start])])
        visited = {start}

        while queue:
            (x, y), path = queue.popleft()

            if (x, y) == end:
                return path

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.maze.width and 0 <= ny < self.maze.height and self.maze.grid[ny][nx] == 0 and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    new_path = list(path)
                    new_path.append((nx, ny))
                    queue.append(((nx, ny), new_path))
        return []

    def update(self):
        if self.path and self.path_index < len(self.path) - 1:
            self.path_index += 1
            self.x, self.y = self.path[self.path_index]

    def draw(self, screen):
        # Draw path
        for i in range(len(self.path) - 1):
            start_pos = (self.path[i][0] * config.CELL_SIZE + config.CELL_SIZE // 2, self.path[i][1] * config.CELL_SIZE + config.CELL_SIZE // 2)
            end_pos = (self.path[i+1][0] * config.CELL_SIZE + config.CELL_SIZE // 2, self.path[i+1][1] * config.CELL_SIZE + config.CELL_SIZE // 2)
            pygame.draw.line(screen, config.RED, start_pos, end_pos, 2)

        # Draw character
        rect = pygame.Rect(self.x * config.CELL_SIZE + (config.CELL_SIZE - config.CHARACTER_SIZE) // 2,
                            self.y * config.CELL_SIZE + (config.CELL_SIZE - config.CHARACTER_SIZE) // 2,
                            config.CHARACTER_SIZE, config.CHARACTER_SIZE)
        pygame.draw.rect(screen, config.CHARACTER_COLOR, rect)

    def has_reached_end(self):
        return (self.x, self.y) == self.maze.end_node
