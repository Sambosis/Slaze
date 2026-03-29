from utils import generate_eat_sound, generate_hit_sound, generate_sink_sound, generate_gameover_sound
import pygame
import random
import math
from board import Board
from snake import Snake
from food import Food
from utils import Particle, draw_ui, save_highscore


class Game:
    """
    Orchestrates game states, updates entities, handles input/events, renders UI/effects,
    manages win/loss/high score for Serpent Seas.
    """
    def __init__(self, high_score: int = 0):
        """
        Initialize the game.

        :param high_score: Loaded high score from main.
        """
        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("Serpent Seas")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 48)
        self.small_font = pygame.font.Font(None, 24)
        self.board = Board()
        self.snake = Snake()
        self.food = Food()
        self.particles = []
        self.score = 0
        self.high_score = high_score
        self.state = 'menu'
        self.paused = False
        self.timer = 0
        self.shake = 0
        try:
            self.bg = pygame.image.load('assets/bg.png').convert()
        except pygame.error:
            self.bg = pygame.Surface((800, 600))
            # Fallback gradient
            for y in range(600):
                ratio = y / 600
                color = (int(0 + 170 * ratio), int(10 + 170 * ratio), int(50 + 200 * ratio))
                pygame.draw.line(self.bg, color, (0, y), (800, y))
        self.sounds = {
            'eat': generate_eat_sound(),
            'hit': generate_hit_sound(),
            'sink': generate_sink_sound(),
            'gameover': generate_gameover_sound()
        }

    def start_game(self):
        """Reset game to playing state."""
        self.state = 'playing'
        self.board.place_ships()
        self.snake = Snake()
        self.food.spawn(self.board, self.snake)
        self.particles = []
        self.score = 0
        self.timer = 0
        self.paused = False
        self.shake = 0

    def restart(self):
        """Restart current game."""
        self.start_game()

    def spawn_particles(self, head_grid: tuple[int, int], count: int, type_: str = 'hit'):
        """
        Spawn particles at grid position.

        :param head_grid: Grid (x, y) position.
        :param count: Number of particles.
        :param type_: Particle type ('eat', 'hit', 'sink').
        """
        px = 150 + head_grid[0] * 50 + 25
        py = 50 + head_grid[1] * 50 + 25
        for _ in range(count):
            self.particles.append(Particle((px, py)))
    def run(self) -> None:
        """Main game loop."""
        if self.state == 'menu':
            pass
        elif self.state == 'playing' and not self.paused:
            collided, new_head = self.snake.update()
            if collided:
                self.state = 'gameover'
                self.sounds['gameover'].play()
                if self.score > self.high_score:
                    self.high_score = self.score
                    save_highscore(self.score)
            elif new_head is not None:
                head = new_head
                ate = self.food.eaten(head)
                is_hit, is_sink = self.board.probe(head[0], head[1])
                if ate:
                    self.score += 10
                    self.snake.grow()
                    self.sounds['eat'].play()
                    self.food.spawn(self.board, self.snake)
                    self.spawn_particles(head, 20, 'eat')
                elif is_hit:
                    self.score += 50 * len(self.snake.body)
                    self.sounds['hit'].play()
                    self.spawn_particles(head, 30, 'hit')
                    if is_sink:
                        self.score += 200
                        self.sounds['sink'].play()
                        self.shake = 15
                        self.spawn_particles(head, 50, 'sink')
                if not ate:
                    self.snake.body.pop(0)
                self.timer += 1
                if self.board.sunk_ships == 4:
                    self.state = 'win'
                    if self.score > self.high_score:
                        self.high_score = self.score
                        save_highscore(self.score)
        
        # Update particles
        self.particles = [p for p in self.particles if p.update(dt)]
        
        # Render
        self.screen.blit(self.bg, (0, 0))
        
        # Shake offset
        sx, sy = 0, 0
        if self.shake > 0:
            sx = random.uniform(-3, 3)
            sy = random.uniform(-3, 3)
            self.shake -= 1
        offset = (150 + sx, 50 + sy)
        self.board.draw(self.screen, offset)
        self.snake.draw(self.screen, offset)
        self.food.draw(self.screen, offset, 50, self.timer * 0.1)
        for p in self.particles:
            p.draw(self.screen)
        draw_ui(self.screen, self)
        
        # State overlays
        if self.state == 'menu':
            title = self.font.render('Serpent Seas', True, (0, 255, 255))
            self.screen.blit(title, (400 - title.get_width() // 2, 200))
            start_text = self.small_font.render('Press SPACE to Start', True, (255, 255, 255))
            self.screen.blit(start_text, (400 - start_text.get_width() // 2, 300))
        elif self.state in ['win', 'gameover']:
            overlay = pygame.Surface((800, 600), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, (0, 0))
            if self.state == 'win':
                title = self.font.render('Victory! Seas Conquered!', True, (0, 255, 0))
                subtitle = self.small_font.render(f'Final Score: {self.score}  Press R to restart or ESC to quit', True, (255, 255, 255))
            else:
                title = self.font.render('Serpent Defeated!', True, (255, 0, 0))
                subtitle = self.small_font.render(f'Final Score: {self.score}  Press R to restart or ESC to quit', True, (255, 255, 255))
            self.screen.blit(title, (400 - title.get_width() // 2, 250))
            self.screen.blit(subtitle, (400 - subtitle.get_width() // 2, 320))
        
        pygame.display.flip()