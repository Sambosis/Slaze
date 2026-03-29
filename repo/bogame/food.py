import pygame
import random
import math


class Food:
    """
    Spawns/eats food on valid cells, pulses glow effect for visuals.
    Attributes:
        pos (tuple[int,int]): Current position (x, y) or None if eaten.
    Methods:
        __init__()
        spawn(board: Board, snake: Snake) -> None
        eaten(snake_head: tuple[int,int]) -> bool
        draw(screen: pygame.Surface, offset=(150,50), cell_size=50, pulse_time=0) -> None
    """

    def __init__(self):
        """
        Initialize food with no position and load sprite image or fallback.
        """
        self.pos = None
        try:
            self.food_img = pygame.image.load('assets/food.png').convert_alpha()
        except pygame.error:
            print("Warning: Could not load food.png, using fallback circle.")
            self.food_img = None

    def spawn(self, board, snake) -> None:
        """
        Spawn food on a random empty non-ship cell, avoiding snake body.
        Tries up to 100 times, then picks first available empty cell.

        :param board: Board instance to check grid
        :param snake: Snake instance to avoid body positions
        """
        attempts = 0
        while attempts < 100:
            x = random.randint(0, board.size - 1)
            y = random.randint(0, board.size - 1)
            if (board.grid[y][x] == 'empty' and
                (x, y) not in snake.body):
                self.pos = (x, y)
                return
            attempts += 1
        # Fallback: find any empty cell not occupied by snake
        for y in range(board.size):
            for x in range(board.size):
                if (board.grid[y][x] == 'empty' and
                    (x, y) not in snake.body):
                    self.pos = (x, y)
                    return
        # Worst case: pick random empty (ignore snake)
        for y in range(board.size):
            for x in range(board.size):
                if board.grid[y][x] == 'empty':
                    self.pos = (x, y)
                    return

    def eaten(self, snake_head: tuple[int, int]) -> bool:
        """
        Check if snake head ate the food. Returns True if eaten, sets pos=None.

        :param snake_head: Position of snake head
        :return: True if position matches food pos
        """
        if self.pos == snake_head:
            self.pos = None
            return True
        return False
    def draw(self, screen: pygame.Surface, offset=(150, 50), cell_size=50, pulse_time=0) -> None:
        """
    Draw pulsing food sprite or fallback glow circle.
    """
        if self.pos:
            px = offset[0] + self.pos[0] * cell_size
            py = offset[1] + self.pos[1] * cell_size
            if self.food_img:
                # Pulsing alpha for sprite
                alpha_surf = self.food_img.copy()
                alpha = 128 + 64 * math.sin(pulse_time)
                alpha_surf.set_alpha(max(0, min(255, int(alpha))))
                screen.blit(alpha_surf, (px, py))
            else:
                # Fallback pulsing cyan glow circle
                cx, cy = px + cell_size // 2, py + cell_size // 2
                pulse = 0.5 + 0.5 * math.sin(pulse_time)
                radius = int(12 + 3 * pulse)
                color = (0, int(255 * pulse), int(255 * pulse))
                pygame.draw.circle(screen, color, (int(cx), int(cy)), radius)