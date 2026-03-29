import pygame


class Snake:
    """
    Handles snake movement, growth, direction queuing, self-collision detection,
    and grid-snapped rendering for the Serpent Seas game.
    """
    def __init__(self, grid_size: int = 10):
        """
        Initialize the snake.

        :param grid_size: Size of the game grid (default 10 for 10x10).
        """
        self.grid_size = grid_size
        self.body = [(4, 4), (5, 4), (6, 4)]  # Tail to head: starts length 3, moving right
        self.direction = (1, 0)  # Current direction (dx, dy)
        self.next_dir = (1, 0)   # Queued next direction
        self.speed = 12          # Frames per move (interval, lower = faster)
        self.move_timer = 0      # Frames until next move
        self.growths = 0         # Count of growths for speed increase

    def change_dir(self, dx: int, dy: int) -> None:
        """
        Queue a direction change, preventing immediate reverse.

        :param dx: Delta x for new direction (-1, 0, 1)
        :param dy: Delta y for new direction (-1, 0, 1)
        """
        new_dir = (dx, dy)
        # Prevent reversing into tail
        opposite = (-self.direction[0], -self.direction[1])
        if new_dir != opposite:
            self.next_dir = new_dir

    def collides(self, x: int, y: int) -> bool:
        """
        Check if position (x, y) overlaps with snake body.

        :param x: Grid x position
        :param y: Grid y position
        :return: True if collides
        """
        return (x, y) in self.body
    def update(self) -> tuple[bool, tuple[int, int] | None]:
        """
    Update snake movement timer and move if ready. Appends new head if valid move.
    Returns (collided: bool, new_head_pos: tuple[int,int] | None)
    - collided True, None: game over (bounds or self-collision)
    - False, pos: successfully moved to new pos
    - False, None: no move yet (timer not ready)
    Popping tail handled externally after eating check.
    """
        self.move_timer += 1
        if self.move_timer < self.speed:
            return False, None
        self.move_timer = 0
        self.direction = self.next_dir
        # Calculate new head position
        head = self.body[-1]
        nx = head[0] + self.direction[0]
        ny = head[1] + self.direction[1]
        # Check bounds
        if not (0 <= nx < self.grid_size and 0 <= ny < self.grid_size):
            return True, None
        # Check self-collision (new head overlaps current body)
        if self.collides(nx, ny):
            return True, None
        # Move: append new head position
        self.body.append((nx, ny))
        return False, (nx, ny)

    def grow(self) -> None:
        """
        Record a growth (from eating food). Adjust speed every 5 growths.
        Length increase handled by not popping tail externally.
        """
        self.growths += 1
        if self.growths % 5 == 0:
            # Progressive speed increase: decrease interval, max speed equiv min interval 4 (20 levels? but cap practical)
            self.speed = max(4, self.speed - 1)

    def draw(self, screen: pygame.Surface, offset=(150, 50), cell_size=50) -> None:
        """
        Draw the snake on screen with gradient (brighter head, darker tail).
        Uses anti-aliased circles for smooth segmented look.

        :param screen: Pygame surface to draw on
        :param offset: (x, y) offset for grid position
        :param cell_size: Pixel size per grid cell
        """
        if not self.body:
            return
        n = len(self.body)
        # Gradient: dark tail (20, 100, 20) to bright head (50, 255, 50)
        tail_color = (20, 100, 20)
        head_color = (50, 255, 50)
        head_radius = 23
        body_radius = 20
        for i, (gx, gy) in enumerate(self.body):
            # Interpolate color: tail dark (i=0), head bright (i=n-1)
            factor = i / max(1, n - 1)
            r = int(tail_color[0] + (head_color[0] - tail_color[0]) * factor)
            g = int(tail_color[1] + (head_color[1] - tail_color[1]) * factor)
            b = int(tail_color[2] + (head_color[2] - tail_color[2]) * factor)
            color = (r, g, b)
            # Center of cell
            cx = offset[0] + gx * cell_size + cell_size // 2
            cy = offset[1] + gy * cell_size + cell_size // 2
            # Draw segment
            radius = head_radius if i == n - 1 else body_radius
            pygame.draw.circle(screen, color, (int(cx), int(cy)), radius)
            # Optional: darker outline for scales effect
            pygame.draw.circle(screen, (0, 50, 0), (int(cx), int(cy)), radius, 1)