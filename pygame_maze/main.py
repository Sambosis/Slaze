
import pygame
import sys
import config
from maze import Maze
from character import Character

def main():
    pygame.init()
    screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
    pygame.display.set_caption(config.SCREEN_TITLE)
    clock = pygame.time.Clock()

    while True:
        maze = Maze(config.MAZE_WIDTH, config.MAZE_HEIGHT)
        maze.generate()
        character = Character(maze.start_node[0], maze.start_node[1], maze)

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        sys.exit()

            character.update()

            screen.fill(config.PATH_COLOR)
            maze.draw(screen)
            character.draw(screen)

            pygame.display.flip()

            if character.has_reached_end():
                running = False

            clock.tick(config.FPS)

if __name__ == '__main__':
    main()