
import pygame
import random
import config

class Maze:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid = [[1 for _ in range(width)] for _ in range(height)]
        self.start_node = (0, 0)
        self.end_node = (width - 1, height - 1)

    def generate(self):
        stack = [(0, 0)]
        self.grid[0][0] = 0

        while stack:
            x, y = stack[-1]
            neighbors = self._get_neighbors(x, y)

            if neighbors:
                nx, ny = random.choice(neighbors)
                self.grid[ny][nx] = 0
                self.grid[(y + ny) // 2][(x + nx) // 2] = 0
                stack.append((nx, ny))
            else:
                stack.pop()
        
        self.grid[self.start_node[1]][self.start_node[0]] = 0
        self.grid[self.end_node[1]][self.end_node[0]] = 0


    def _get_neighbors(self, x, y):
        neighbors = []
        for dx, dy in [(0, -2), (0, 2), (-2, 0), (2, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height and self.grid[ny][nx] == 1:
                neighbors.append((nx, ny))
        return neighbors

    def draw(self, screen):
        for y, row in enumerate(self.grid):
            for x, cell in enumerate(row):
                if cell == 1:
                    pygame.draw.rect(screen, config.WALL_COLOR, (x * config.CELL_SIZE, y * config.CELL_SIZE, config.CELL_SIZE, config.CELL_SIZE))
        
        # Draw start and end points
        start_rect = pygame.Rect(self.start_node[0] * config.CELL_SIZE, self.start_node[1] * config.CELL_SIZE, config.CELL_SIZE, config.CELL_SIZE)
        end_rect = pygame.Rect(self.end_node[0] * config.CELL_SIZE, self.end_node[1] * config.CELL_SIZE, config.CELL_SIZE, config.CELL_SIZE)
        pygame.draw.rect(screen, config.START_COLOR, start_rect)
        pygame.draw.rect(screen, config.END_COLOR, end_rect)


