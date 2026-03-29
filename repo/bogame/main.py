import sys
import pygame
from game import Game
from utils import load_highscore, save_highscore


def main() -> None:
    """
    Main entry point for Serpent Seas game.
    Initializes Pygame and mixer, loads high score, creates and runs Game instance.
    Handles clean shutdown.
    """
    # Pre-initialize mixer for procedural sounds (22050Hz, 16-bit signed, mono, 512 buffer)
    pygame.mixer.pre_init(22050, -16, 1, 512)
    pygame.init()
    
    # Load persistent high score
    high_score = load_highscore()
    
    # Create and run the game
    game = Game(high_score)
    game.run()
    
    # Clean shutdown
    pygame.quit()
    sys.exit()


if __name__ == '__main__':
    main()