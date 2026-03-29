import pygame
import sys
from maze import Maze, generate_maze
from character import Character, find_path
from renderer import draw_maze

def main() -> None:
    """
    Main game loop that initializes Pygame, generates mazes, and handles the game state.
    The character autonomously navigates through the maze, and the maze regenerates
    when the character reaches the end. The user can exit by pressing ESC.
    """
    # Initialize Pygame
    pygame.init()

    # Constants
    SCREEN_WIDTH = 800
    SCREEN_HEIGHT = 800
    MAZE_WIDTH = 21  # Odd number for proper maze generation
    MAZE_HEIGHT = 21
    FPS = 60
    CHARACTER_SPEED = 4.0  # cells per second

    # Set up the display
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Maze Navigation")

    # Clock for controlling frame rate
    clock = pygame.time.Clock()

    # Generate initial maze
    maze = generate_maze(MAZE_WIDTH, MAZE_HEIGHT)

    # Create character at the start position
    character = Character(maze.start, CHARACTER_SPEED)

    # Find initial path
    path = find_path(maze, maze.start, maze.end)
    if not path:
        print("Error: No valid path found for initial maze!")
        pygame.quit()
        sys.exit(1)
    character.set_path(path)

    # Main game loop
    running = True
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        # Calculate delta time for smooth movement
        dt = clock.tick(FPS) / 1000.0  # Convert milliseconds to seconds

        # Update character position
        character.update(dt)

        # Check if character reached the end
        if character.has_reached_destination():
            # Generate new maze
            maze = generate_maze(MAZE_WIDTH, MAZE_HEIGHT)

            # Reset character position
            character = Character(maze.start, CHARACTER_SPEED)

            # Find new path
            path = find_path(maze, maze.start, maze.end)
            if not path:
                print("Error: No valid path found for new maze!")
                pygame.quit()
                sys.exit(1)
            character.set_path(path)

        # Draw everything
        draw_maze(screen, maze, character)

        # Update display
        pygame.display.flip()

    # Clean up
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()