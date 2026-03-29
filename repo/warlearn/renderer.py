import pygame
import numpy as np
import time
from typing import Dict, Any, Optional
from env import Board


class Renderer:
    def __init__(self, screen_size: tuple = (800, 700)):
        pygame.init()
        self.screen = pygame.display.set_mode(screen_size)
        pygame.display.set_caption("War Game AI Trainer")
        self.tile_size = 50
        self.board_rect = pygame.Rect(100, 50, 600, 600)
        self.font = pygame.font.Font(None, 24)
        self.big_font = pygame.font.Font(None, 48)
        self.save_requested = False

    def handle_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_s:
                    self.save_requested = True
        return True

    def get_tile_rect(self, r: float, c: float) -> pygame.Rect:
        x = self.board_rect.x + int(c * self.tile_size)
        y = self.board_rect.y + int(r * self.tile_size)
        return pygame.Rect(x, y, self.tile_size, self.tile_size)

    def draw_terrain(self, board: Board) -> None:
        for ri in range(12):
            for ci in range(12):
                rect = self.get_tile_rect(ri, ci)
                if board.terrain[ri, ci] == 0:
                    geo = int(30 * np.sin(ri / 12 * np.pi))
                    color = (0, 180 + geo, 50)
                    pygame.draw.rect(self.screen, color, rect)
                else:
                    base_color = (95, 90, 80)
                    dark_color = (70, 65, 60)
                    pygame.draw.rect(self.screen, base_color, rect)
                    bottom_half_rect = pygame.Rect(rect.left, rect.centery, rect.w, rect.h // 2)
                    pygame.draw.rect(self.screen, dark_color, bottom_half_rect)
                    pygame.draw.line(self.screen, (105, 100, 90), (rect.centerx, rect.top + 5), (rect.centerx, rect.bottom - 5), 2)
                pygame.draw.rect(self.screen, (40, 40, 40), rect, 1)

    def draw_unit(self, rect: pygame.Rect, is_p1: bool, hp: int) -> None:
        center = rect.center
        size = min(rect.w // 2, rect.h // 2) - 5
        color = (220, 80, 80) if is_p1 else (80, 80, 220)
        glow_radius = size + 4
        glow_surf_size = glow_radius * 2
        glow_surf = pygame.Surface((glow_surf_size, glow_surf_size), pygame.SRCALPHA)
        glow_center = (glow_surf_size // 2, glow_surf_size // 2)
        pygame.draw.circle(glow_surf, (*color, 80), glow_center, glow_radius)
        glow_pos = (center[0] - glow_surf_size // 2, center[1] - glow_surf_size // 2)
        self.screen.blit(glow_surf, glow_pos)
        pygame.draw.circle(self.screen, color, center, size)
        armor_rect = pygame.Rect(center[0] - size // 2, center[1] - size // 2, size, size // 2)
        pygame.draw.rect(self.screen, (160, 160, 160), armor_rect)
        bar_rect = pygame.Rect(center[0] - 15, rect.top - 8, 30, 6)
        pygame.draw.rect(self.screen, (255, 50, 50), bar_rect)
        fill_rect = bar_rect.copy()
        fill_rect.width = int(fill_rect.width * (hp / 10.0))
        if fill_rect.width > 0:
            pygame.draw.rect(self.screen, (50, 255, 50), fill_rect)
        pygame.draw.rect(self.screen, (0, 0, 0), bar_rect, 1)

    def _draw_overlays(self, board: Board, stats: Optional[Dict[str, Any]], clock: pygame.time.Clock) -> None:
        turn_surf = self.font.render(f'Turn {board.turn}', True, (255, 255, 255))
        self.screen.blit(turn_surf, (110, 650))
        done, winner = board.is_done()
        if done:
            if winner == 'p1_win':
                color = (255, 200, 0)
                text = 'Red Wins!'
            elif winner == 'p2_win':
                color = (0, 200, 255)
                text = 'Blue Wins!'
            else:
                color = (200, 200, 200)
                text = 'Draw!'
            text_surf = self.big_font.render(text, True, color)
            text_rect = text_surf.get_rect(center=(self.board_rect.centerx, self.board_rect.centery))
            ticks = pygame.time.get_ticks()
            shake_x = int(3 * np.sin(ticks * 0.01))
            self.screen.blit(text_surf, (text_rect.x + shake_x, text_rect.y))
        if stats:
            y = 20
            for stat_key in ['episodes', 'avg_reward', 'winrate']:
                if stat_key in stats:
                    val = stats[stat_key]
                    if isinstance(val, float):
                        text = f'{stat_key.replace("_", " ").title()}: {val:.2f}'
                    else:
                        text = f'{stat_key.replace("_", " ").title()}: {val}'
                    text_surf = self.font.render(text, True, (255, 255, 255))
                    self.screen.blit(text_surf, (110, y))
                    y += 25
        fps_surf = self.font.render(f'FPS: {clock.get_fps():.1f}', True, (255, 255, 255))
        self.screen.blit(fps_surf, (650, 20))

    def render(self, board: Board, clock: pygame.time.Clock, step_delay: float = 0.2, stats: Optional[Dict[str, Any]] = None) -> bool:
        """Renders the board with animations if active, handles events, displays overlays and stats. Returns False if quit requested."""
        if not self.handle_events():
            return False
        anim_dur = getattr(board, 'anim_duration', 0.01)
        unit_anims = getattr(board, 'unit_anims', {})
        flash_tiles = getattr(board, 'flash_tiles', [])
        anim_t = getattr(board, 'anim_t', 0.0)
        while anim_t < anim_dur:
            dt = clock.tick(30) / 1000.0
            if not self.handle_events():
                return False
            anim_t = min(anim_dur, anim_t + dt)
            board.anim_t = anim_t
            t = anim_t / anim_dur
            self.screen.fill((15, 40, 20))
            self.draw_terrain(board)
            all_units = board.p1_units + board.p2_units
            for unit in all_units:
                uid = id(unit)
                if uid in unit_anims:
                    prev, target = unit_anims[uid]
                    lerp_r = prev[0] + t * (target[0] - prev[0])
                    lerp_c = prev[1] + t * (target[1] - prev[1])
                else:
                    lerp_r, lerp_c = unit.pos
                rect = self.get_tile_rect(lerp_r, lerp_c)
                self.draw_unit(rect, unit in board.p1_units, unit.hp)
            if t > 0.5:
                flash_t = (t - 0.5) / 0.5
                alpha = min(255, int(255 * flash_t))
                for pos in flash_tiles:
                    rect = self.get_tile_rect(*pos)
                    flash_surf = pygame.Surface(rect.size, pygame.SRCALPHA)
                    flash_surf.fill((255, 255, 0, alpha))
                    self.screen.blit(flash_surf, rect.topleft)
            self._draw_overlays(board, stats, clock)
            pygame.display.flip()
        if not self.handle_events():
            return False
        self.screen.fill((15, 40, 20))
        self.draw_terrain(board)
        all_units = board.p1_units + board.p2_units
        for unit in all_units:
            rect = self.get_tile_rect(*unit.pos)
            self.draw_unit(rect, unit in board.p1_units, unit.hp)
        self._draw_overlays(board, stats, clock)
        pygame.display.flip()
        time.sleep(step_delay)
        return True