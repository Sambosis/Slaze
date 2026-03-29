import pygame
from maze import Maze
from character import Character

# Color definitions
WALL_COLOR = (40, 40, 40)        # Dark gray walls
PATH_COLOR = (255, 255, 255)     # White paths
START_COLOR = (0, 255, 0)        # Green start position
END_COLOR = (255, 0, 0)          # Red end position
CHARACTER_COLOR = (0, 120, 255)  # Blue character
BORDER_COLOR = (0, 0, 0)         # Black borders
HIGHLIGHT_COLOR = (255, 255, 200) # Character highlight

def draw_maze(screen: pygame.Surface, maze: Maze, character: Character) -> None:
    """
    Renders the maze and character on the given Pygame surface.
    """
    # Validate input types
    if not isinstance(screen, pygame.Surface):
        raise TypeError(f"Expected 'screen' to be a pygame.Surface, got {type(screen).__name__}")
    if not isinstance(maze, Maze):
        raise TypeError(f"Expected 'maze' to be a Maze, got {type(maze).__name__}")
    if not isinstance(character, Character):
        raise TypeError(f"Expected 'character' to be a Character, got {type(character).__name__}")

    # Calculate cell dimensions
    screen_width, screen_height = screen.get_size()
    cell_width = screen_width // maze.width
    cell_height = screen_height // maze.height
    cell_size = min(cell_width, cell_height)

    # Calculate offset to center the maze
    maze_width = maze.width * cell_size
    maze_height = maze.height * cell_size
    offset_x = (screen_width - maze_width) // 2
    offset_y = (screen_height - maze_height) // 2

    # Clear screen with background color
    screen.fill((30, 30, 30))

    # Draw maze grid
    for y in range(maze.height):
        for x in range(maze.width):
            cell_rect = pygame.Rect(
                offset_x + x * cell_size,
                offset_y + y * cell_size,
                cell_size,
                cell_size
            )

            # Determine cell color
            if (x, y) == maze.start:
                pygame.draw.rect(screen, START_COLOR, cell_rect)
            elif (x, y) == maze.end:
                pygame.draw.rect(screen, END_COLOR, cell_rect)
            elif maze.grid[y][x] == 1:  # Wall
                pygame.draw.rect(screen, WALL_COLOR, cell_rect)
            else:  # Path
                pygame.draw.rect(screen, PATH_COLOR, cell_rect)

            # Draw cell borders
            pygame.draw.rect(screen, BORDER_COLOR, cell_rect, 1)

    # Draw character with highlight
    char_x, char_y = character.position
    char_center = (
        offset_x + char_x * cell_size + cell_size // 2,
        offset_y + char_y * cell_size + cell_size // 2
    )
    char_radius = cell_size // 3

    pygame.draw.circle(screen, CHARACTER_COLOR, char_center, char_radius)
    pygame.draw.circle(screen, HIGHLIGHT_COLOR, char_center, char_radius // 2)