import pygame
import math
import random
import os
import struct

class Particle:
    """
    Individual particle for explosion/smoke effects on hits/sinks.
    """
    def __init__(self, pos: tuple[float, float]):
        """
        Initialize a particle at position with random velocity, life, color, and size.
        """
        self.pos = list(pos)
        # Random velocity: outward and slightly upward for sparks
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(1.0, 3.0)
        self.vel = (math.cos(angle) * speed, math.sin(angle) * speed - 1.0)
        self.life = random.uniform(0.4, 0.8)
        self.max_life = self.life
        # Orange sparks
        self.color = (
            random.randint(255, 255),
            random.randint(100, 200),
            random.randint(0, 50),
            255
        )
        self.size = random.uniform(2.0, 6.0)

    def update(self, dt: float) -> bool:
        """
        Update particle position, life, and fade color alpha.
        Returns True if still alive.
        """
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]
        self.life -= dt
        if self.life > 0:
            alpha_ratio = self.life / self.max_life
            self.color = (
                self.color[0],
                self.color[1],
                self.color[2],
                int(255 * alpha_ratio)
            )
            # Shrink size as it fades
            self.size *= 0.98
            return True
        return False

    def draw(self, screen: pygame.Surface) -> None:
        """
        Draw the particle as an anti-aliased circle.
        """
        color = self.color
        pos = (int(self.pos[0]), int(self.pos[1]))
        size = max(1, int(self.size))
        if size > 0:
            pygame.draw.circle(screen, color, pos, size)

def load_highscore(filename: str = 'highscore.txt') -> int:
    """
    Loads high score from file, returns 0 if missing or invalid.
    """
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                return int(f.read().strip())
        except (ValueError, IOError):
            pass
    return 0

def save_highscore(score: int, filename: str = 'highscore.txt') -> None:
    """
    Saves high score to file if it is higher than current.
    """
    hs = load_highscore(filename)
    if score > hs:
        try:
            with open(filename, 'w') as f:
                f.write(str(score))
        except IOError:
            pass
def draw_ui(screen: pygame.Surface, game) -> None:
    """
    Renders score, high score, ship status bars, pause overlay, instructions.
    Assumes game has font, small_font, score, high_score, board.ships (in fixed order),
    board.sunk_ships, paused, state.
    """
    # Top bar: 0-800 x 0-50
    pygame.draw.rect(screen, (0, 20, 50), (0, 0, 800, 50))
    minutes = game.timer // 60
    seconds = game.timer % 60
    score_text = game.font.render(f"Score: {game.score}  High Score: {game.high_score}  Time: {minutes}:{seconds:02d}  Sunk: {game.board.sunk_ships}/4", True, (255, 255, 255))
    screen.blit(score_text, (10, 10))

    # Left panel ship status: 0-150 x 50-550
    pygame.draw.rect(screen, (0, 50, 100), (0, 50, 150, 500))
    ship_data = [
        ("Destroyer", (255, 165, 0), 2),
        ("Light Cruiser", (100, 150, 255), 3),
        ("Heavy Cruiser", (0, 150, 255), 3),
        ("Battleship", (200, 0, 0), 4)
    ]
    y_offset = 70
    for i, (name, color, length) in enumerate(ship_data):
        if i < len(game.board.ships):
            ship = game.board.ships[i]
            hits_ratio = ship.hits / length
            # Ship icon (small rect)
            pygame.draw.rect(screen, color, (20, y_offset, 20, 12))
            # Progress bar
            bar_rect = pygame.Rect(50, y_offset + 5, 90, 10)
            pygame.draw.rect(screen, (50, 50, 50), bar_rect)
            fill_rect = pygame.Rect(50, y_offset + 5, int(90 * hits_ratio), 10)
            pygame.draw.rect(screen, (0, 255, 0) if not ship.sunk else (255, 0, 0), fill_rect)
            pygame.draw.rect(screen, (255, 255, 255), bar_rect, 1)
            # Sunk label
            if ship.sunk:
                sunk_text = game.small_font.render("SUNK", True, (255, 255, 0))
                screen.blit(sunk_text, (bar_rect.centerx - sunk_text.get_width() // 2, y_offset + 18))
            y_offset += 60

    # Bottom instructions: 0-800 x 550-600
    pygame.draw.rect(screen, (0, 20, 50), (0, 550, 800, 50))
    instr_text = game.small_font.render("Arrows: move  Space: pause  R: restart  ESC: quit", True, (255, 255, 255))
    text_rect = instr_text.get_rect(center=(400, 575))
    screen.blit(instr_text, text_rect)

    # Pause overlay
    if game.paused:
        overlay = pygame.Surface((800, 600), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 128))
        screen.blit(overlay, (0, 0))
        pause_text = game.font.render("PAUSED", True, (255, 255, 255))
        pause_rect = pause_text.get_rect(center=(400, 300))
        screen.blit(pause_text, pause_rect)

def draw_gradient(screen: pygame.Surface, rect: pygame.Rect, start_color: tuple[int, int, int], end_color: tuple[int, int, int]) -> None:
    """
    Draws a vertical gradient background.
    """
    for y in range(rect.bottom - rect.top):
        ratio = y / (rect.bottom - rect.top)
        r = int(start_color[0] * (1 - ratio) + end_color[0] * ratio)
        g = int(start_color[1] * (1 - ratio) + end_color[1] * ratio)
        b = int(start_color[2] * (1 - ratio) + end_color[2] * ratio)
        pygame.draw.line(screen, (r, g, b), (rect.left, rect.top + y), (rect.right, rect.top + y))

def generate_eat_sound() -> pygame.mixer.Sound:
    """
    Generates a short high beep (800Hz sine wave, 0.1s, mono 22050Hz).
    """
    sample_rate = 22050
    duration = 0.1
    freq = 800
    amp = 0.3
    samples_count = int(sample_rate * duration)
    samples = [int(amp * 32767 * math.sin(2 * math.pi * freq * i / sample_rate)) for i in range(samples_count)]
    sound_data = struct.pack(f'>{samples_count}h', *samples)
    return pygame.mixer.Sound(buffer=sound_data)

def generate_hit_sound() -> pygame.mixer.Sound:
    """
    Generates a low pulse boom (200Hz sine wave with envelope, 0.2s).
    """
    sample_rate = 22050
    duration = 0.2
    freq = 200
    amp = 0.4
    samples_count = int(sample_rate * duration)
    samples = []
    for i in range(samples_count):
        t = i / sample_rate
        # Simple ADSR envelope
        env = 1.0
        if t < 0.01:  # attack
            env = t / 0.01
        elif t > duration - 0.05:  # release
            env = (duration - t) / 0.05
        sample = int(amp * 32767 * env * math.sin(2 * math.pi * freq * t))
        samples.append(sample)
    sound_data = struct.pack(f'>{samples_count}h', *samples)
    return pygame.mixer.Sound(buffer=sound_data)

def generate_sink_sound() -> pygame.mixer.Sound:
    """
    Generates a rising fanfare chord (multiple freqs chirp up, 0.5s).
    """
    sample_rate = 22050
    duration = 0.5
    freq_start = 220
    freq_end = 800
    amps = [0.2, 0.15, 0.1]  # chord volumes
    freqs_base = [1.0, 1.5, 2.0]  # harmonic
    samples_count = int(sample_rate * duration)
    samples = []
    for i in range(samples_count):
        t = i / sample_rate
        freq = freq_start + (freq_end - freq_start) * (t / duration)
        sample_val = 0
        for j, mult in enumerate(freqs_base):
            wave = math.sin(2 * math.pi * freq * mult * t)
            sample_val += amps[j] * wave
        sample_val = min(1.0, max(-1.0, sample_val)) * 32767
        samples.append(int(sample_val))
    sound_data = struct.pack(f'>{samples_count}h', *samples)
    return pygame.mixer.Sound(buffer=sound_data)

def generate_gameover_sound() -> pygame.mixer.Sound:
    """
    Generates a low ominous drone (100Hz decaying, 1.0s).
    """
    sample_rate = 22050
    duration = 1.0
    freq = 100
    amp = 0.2
    samples_count = int(sample_rate * duration)
    samples = []
    for i in range(samples_count):
        t = i / sample_rate
        env = math.exp(-t * 3)  # decay
        sample = int(amp * 32767 * env * math.sin(2 * math.pi * freq * t))
        samples.append(sample)
    sound_data = struct.pack(f'>{samples_count}h', *samples)
    return pygame.mixer.Sound(buffer=sound_data)