import pygame
import sys
import random
import math
from game import Game
from renderer import Renderer

def main():
    """Main entry point: initializes Pygame, Game, Renderer, and runs the event/render loop."""
    pygame.init()
    screen = pygame.display.set_mode((1024, 768))
    pygame.display.set_caption('AI War Game')
    clock = pygame.time.Clock()
    game = Game(screen, clock)
    renderer = Renderer(screen)
    game.renderer = renderer
    
    # Fonts for victory screen
    font_victory = pygame.font.SysFont('arial', 72, bold=True)
    font_subtitle = pygame.font.SysFont('arial', 36, bold=False)
    
    running = True
    victory_particles = []
    
    while running:
        dt = clock.tick(60) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    game.perform_ai_turn()

        game.update(dt)

        winner = game.check_win()
        if winner is not None:
            # Victory screen with fireworks
            screen.fill((15, 15, 30))  # Dark blue-ish background

            # Victory text
            title_text = font_victory.render(f"{winner.upper()} WINS!", True, (255, 255, 255))
            title_rect = title_text.get_rect(center=(512, 300))
            screen.blit(title_text, title_rect)

            subtitle_text = font_subtitle.render("Press SPACE or ESC to quit", True, (200, 200, 200))
            subtitle_rect = subtitle_text.get_rect(center=(512, 400))
            screen.blit(subtitle_text, subtitle_rect)

            # Initialize fireworks if needed
            if not victory_particles:
                for _ in range(150):
                    victory_particles.append({
                        'pos': [512.0, 384.0],
                        'vel': [random.uniform(-400, 400), random.uniform(-500, -100)],
                        'life': random.uniform(0.8, 1.2),
                        'max_life': random.uniform(0.8, 1.2),
                        'color': random.choice([
                            (255, 50, 50), (255, 150, 50), (255, 255, 50),
                            (100, 255, 100), (50, 100, 255), (200, 50, 255)
                        ]),
                        'size': random.uniform(2, 6)
                    })

            # Update and draw particles
            for particle in victory_particles[:]:
                particle['pos'][0] += particle['vel'][0] * dt
                particle['pos'][1] += particle['vel'][1] * dt
                particle['vel'][0] *= 0.97
                particle['vel'][1] *= 0.97
                particle['life'] -= dt * 1.5
                if particle['life'] <= 0:
                    victory_particles.remove(particle)
                    continue

                life_ratio = particle['life'] / particle['max_life']
                col = tuple(int(c * life_ratio) for c in particle['color'])
                size = int(particle['size'] * life_ratio)
                if size > 0:
                    pos = (int(particle['pos'][0]), int(particle['pos'][1]))
                    pygame.draw.circle(screen, col, pos, size)
                    # Trail effect
                    pygame.draw.circle(screen, (*col, 0), pos, size + 2)

            pygame.display.flip()

            # Poll for exit input
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_SPACE, pygame.K_ESCAPE):
                        running = False
            continue

        # Normal game rendering
        renderer.draw_board(screen, game.board)
        renderer.draw_units(screen, game.units, dt)
        renderer.draw_ui(screen, game)

        pygame.display.flip()
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    random.seed(42)  # For replayability (though set in Board too)
    main()