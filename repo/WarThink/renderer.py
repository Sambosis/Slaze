import pygame
from game import GameState, Unit
from typing import Tuple, List, Dict, Optional

class Renderer:
    def __init__(self, width: int = 1024, height: int = 768):
        pygame.init()
        self.screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
        pygame.display.set_caption('WarThink AI - Asymmetric Self-Play PPO')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.big_font = pygame.font.Font(None, 48)
        self.tile_size = 32
        self.iso_offset_x = width // 2
        self.iso_offset_y = height // 2 - 100
        self.animations: List[Dict] = []
        self.fullscreen = False
        self._update_offsets_and_tile()

    def _update_offsets_and_tile(self):
        w, h = self.screen.get_size()
        self.iso_offset_x = w // 2
        self.iso_offset_y = h // 2 - 100
        self.tile_size = min(w // 20, h // 15)
        self.tile_size = max(24, self.tile_size - (self.tile_size % 4))

    def toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        flags = pygame.FULLSCREEN if self.fullscreen else pygame.RESIZABLE
        size = (0, 0) if self.fullscreen else (1024, 768)
        self.screen = pygame.display.set_mode(size, flags)
        self._update_offsets_and_tile()

    def iso_to_screen(self, grid_pos: Tuple[int, int]) -> Tuple[float, float]:
        row, col = grid_pos
        x = (col - row) * self.tile_size // 2 + self.iso_offset_x
        y = (col + row) * self.tile_size // 4 + self.iso_offset_y
        return x, y

    def draw_grid(self):
        self.screen.fill((135, 206, 235))
        for row in range(10):
            for col in range(10):
                x, y = self.iso_to_screen((row, col))
                color = (34, 139, 34) if (row + col) % 2 == 0 else (139, 69, 19)
                points = [
                    (x, y),
                    (x + self.tile_size // 2, y - self.tile_size // 4),
                    (x + self.tile_size, y),
                    (x + self.tile_size // 2, y + self.tile_size // 4)
                ]
                pygame.draw.polygon(self.screen, color, points)
                pygame.draw.polygon(self.screen, (0, 0, 0), points, 1)

    def draw_unit_pos(self, x: float, y: float, unit: Unit):
        color = (0, 100, 255) if unit.player == 1 else (255, 0, 100)
        size = self.tile_size // 2

        # Glow
        glow_surf = pygame.Surface((size * 2 + 8, size * 2 + 8), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*color, 80), (size + 4, size + 4), size + 4)
        self.screen.blit(glow_surf, (x - size - 4, y - size - 4))

        # Shadow
        shadow_surf = pygame.Surface((size + 10, size // 2 + 8), pygame.SRCALPHA)
        pygame.draw.ellipse(shadow_surf, (0, 0, 0, 120), (3, 3, size + 2, size // 2))
        self.screen.blit(shadow_surf, (x - size // 2 + 4, y - size // 4 + 4))

        # Body
        points = [
            (x, y - size // 2),
            (x + size // 2, y),
            (x, y + size // 2),
            (x - size // 2, y)
        ]
        pygame.draw.polygon(self.screen, color, points)
        pygame.draw.polygon(self.screen, (255, 255, 255), points, 2)

        # Type icon
        type_color = (255, 215, 0) if unit.type_ == 'commander' else (255, 255, 255)
        pygame.draw.circle(self.screen, type_color, (int(x), int(y)), 5)

        # HP bar
        bar_width = 24
        bar_height = 4
        hp_ratio = unit.hp / unit.max_hp
        bar_x = x - bar_width // 2
        bar_y = y - size // 2 - 10
        pygame.draw.rect(self.screen, (200, 0, 0), (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, (0, 200, 0), (bar_x, bar_y, int(bar_width * hp_ratio), bar_height))
        pygame.draw.rect(self.screen, (255, 255, 255), (bar_x, bar_y, bar_width, bar_height), 1)

    def draw_unit(self, unit: Unit):
        x, y = self.iso_to_screen(unit.pos)
        self.draw_unit_pos(x, y, unit)

    def animate_move(self, unit: Unit, target_pos: Tuple[int, int], duration: float = 0.5):
        self.animations.append({
            'unit': unit,
            'start': self.iso_to_screen(unit.pos),
            'end': self.iso_to_screen(target_pos),
            'progress': 0.0,
            'duration': duration
        })

    def update_animations(self, dt: float):
        for anim in self.animations[:]:
            anim['progress'] += dt / anim['duration']
            if anim['progress'] >= 1.0:
                self.animations.remove(anim)
                continue
            prog = 1.0 - (1.0 - anim['progress']) ** 2
            cx = anim['start'][0] + (anim['end'][0] - anim['start'][0]) * prog
            cy = anim['start'][1] + (anim['end'][1] - anim['start'][1]) * prog
            self.draw_unit_pos(cx, cy, anim['unit'])

    def render(self, state: GameState):
        self.draw_grid()
        self.update_animations(1 / 60.0)
        for units in [state.p1_units, state.p2_units]:
            for unit in units:
                if unit.alive:
                    animating = any(a['unit'] == unit for a in self.animations)
                    if not animating:
                        self.draw_unit(unit)
        # UI
        turn_text = self.font.render(
            f'Turn: {state.turn_count} | Player: {"P1 (Blue)" if state.current_player == 1 else "P2 (Red)"}',
            True, (255, 255, 255)
        )
        self.screen.blit(turn_text, (10, 10))
        if state.winner is not None:
            if state.winner == 0:
                win_text = self.big_font.render('DRAW!', True, (255, 255, 0))
            else:
                win_text = self.big_font.render(f'Player {state.winner} WINS!', True, (255, 255, 0))
            tw, th = win_text.get_size()
            self.screen.blit(win_text, (self.screen.get_width() // 2 - tw // 2, self.screen.get_height() // 2 - th // 2))
        pygame.display.flip()

    def draw_stats(self, episode: int, avg_reward: float, win_rate: float, turn_count: Optional[int] = None, winner: Optional[int] = None):
        stats = [
            f'Episode: {episode}',
            f'Avg Reward: {avg_reward:.2f}',
            f'P1 Win Rate: {win_rate:.1%}',
        ]
        if turn_count is not None:
            stats.append(f'Eval Turn: {turn_count}')
        if winner is not None:
            wtext = 'DRAW' if winner == 0 else f'P{winner}'
            stats.append(f'Winner: {wtext}')
        for i, s in enumerate(stats):
            color = (0, 255, 0) if 'Win Rate' in s and win_rate > 0.5 else (255, 255, 255)
            text = self.font.render(s, True, color)
            self.screen.blit(text, (10, 50 + i * 25))

    def handle_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                if event.key == pygame.K_F11:
                    self.toggle_fullscreen()
            elif event.type == pygame.VIDEORESIZE:
                self._update_offsets_and_tile()
        return True