import pygame
import math
import random
from game import Board, Unit, Game

class Renderer:
    """Handles all visual rendering, animations, particles, and UI elements using Pygame drawing."""

    def __init__(self, screen):
        self.screen = screen
        self.window_size = screen.get_size()  # (1024, 768)
        self.board_offset = ((1024 - 720) // 2, (768 - 480) // 2)  # (152, 144)
        self.tile_size = 48
        self.particles = []
        self.shake_timer = 0.0
        self.shake_intensity = 0.0
        self.shake_x = 0.0
        self.shake_y = 0.0
        self.bg_surf = self._create_gradient()
        self.font_title = pygame.font.SysFont('arial', 48, bold=True)
        self.font_ui = pygame.font.SysFont('arial', 24, bold=True)
        self.font_small = pygame.font.SysFont('arial', 18)
        self.TERRAIN_COLORS = {
            'grass': (76, 175, 80),
            'forest': (46, 125, 50),
            'mountain': (121, 85, 72),
            'river': (33, 150, 243)
        }
        self.PLAYER_COLORS = {
            'red': (196, 30, 58),
            'blue': (30, 58, 196)
        }

    def _create_gradient(self):
        """Create a full-window vertical gradient background from sky blue to ground green."""
        surf = pygame.Surface(self.window_size).convert()
        height = self.window_size[1]
        sky = (135, 206, 235)
        ground = (34, 139, 34)
        for y in range(height):
            ratio = y / height
            color = tuple(int(sky[i] * (1 - ratio) + ground[i] * ratio) for i in range(3))
            pygame.draw.line(surf, color, (0, y), (self.window_size[0], y))
        return surf

    def _ease_in_out(self, t):
        """Smooth ease-in-out interpolation (cubic hermite)."""
        return t * t * (3.0 - 2.0 * t)

    def trigger_shake(self, duration=0.5):
        """Trigger a screen shake effect for combat feedback."""
        self.shake_timer = max(self.shake_timer, duration)
        self.shake_intensity = 12.0

    def _update_shake(self, dt):
        """Update shake timer and compute current offsets."""
        if self.shake_timer > 0:
            self.shake_timer -= dt
            t = pygame.time.get_ticks() / 1000.0
            decay = min(1.0, self.shake_timer / 2.0)
            self.shake_x = math.sin(t * 20) * self.shake_intensity * decay
            self.shake_y = math.cos(t * 20 + 0.5) * self.shake_intensity * decay
        else:
            self.shake_x = 0.0
            self.shake_y = 0.0

    def add_explosion(self, board_pos):
        """Add explosion particles at board position (called from game on combat)."""
        screen_pos = (
            self.board_offset[0] + board_pos[0] * self.tile_size + self.tile_size // 2,
            self.board_offset[1] + board_pos[1] * self.tile_size + self.tile_size // 2
        )
        num_particles = random.randint(10, 20)
        for _ in range(num_particles):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(100, 300)
            self.particles.append({
                'pos': [screen_pos[0], screen_pos[1]],
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': random.uniform(0.6, 1.2),
                'max_life': 1.0,
                'color': (random.randint(200, 255), random.randint(50, 150), random.randint(0, 50)),
                'size': random.uniform(2, 6)
            })

    def animate_explosion(self, pos, dt):
        """Update explosion particles and return active ones (for external use if needed)."""
        self._update_particles(dt)
        return [p for p in self.particles if p['life'] > 0]

    def _update_particles(self, dt):
        """Internal update for particles."""
        for particle in self.particles[:]:
            particle['pos'][0] += particle['vel'][0] * dt
            particle['pos'][1] += particle['vel'][1] * dt
            particle['vel'][0] *= 0.96
            particle['vel'][1] *= 0.96
            particle['life'] -= dt
            if particle['life'] <= 0:
                self.particles.remove(particle)

    def draw_board(self, screen, board):
        """Draw the board with terrain, shadows, ownership overlays, and HQ flags."""
        screen.blit(self.bg_surf, (0, 0))

        shake_offset = (self.shake_x, self.shake_y)
        offset = (self.board_offset[0] + shake_offset[0], self.board_offset[1] + shake_offset[1])

        for x in range(board.width):
            for y in range(board.height):
                tile = board.grid[x][y]
                rect = pygame.Rect(
                    offset[0] + x * self.tile_size,
                    offset[1] + y * self.tile_size,
                    self.tile_size,
                    self.tile_size
                )
                color = self.TERRAIN_COLORS[tile.terrain]

                # Tile rect
                pygame.draw.rect(screen, color, rect)

                # Shadow overlay bottom-right
                shadow_surf = pygame.Surface((self.tile_size, self.tile_size), pygame.SRCALPHA)
                shadow_color = tuple(int(c * 0.7) for c in color)
                shadow_rect = pygame.Rect(self.tile_size // 2, self.tile_size // 2, self.tile_size // 2, self.tile_size // 2)
                pygame.draw.rect(shadow_surf, shadow_color + (128,), shadow_rect)
                screen.blit(shadow_surf, rect.topleft)

                # Owner overlay
                if tile.owner != 'neutral':
                    overlay_surf = pygame.Surface((self.tile_size, self.tile_size), pygame.SRCALPHA)
                    overlay_color = self.PLAYER_COLORS[tile.owner]
                    pygame.draw.rect(overlay_surf, (*overlay_color, 51), (0, 0, self.tile_size, self.tile_size))  # ~20% alpha
                    screen.blit(overlay_surf, rect.topleft)

                # HQ flag
                if tile.hq:
                    flag_color = self.PLAYER_COLORS[tile.owner]
                    # Pole
                    pole_rect = pygame.Rect(rect.right - 4, rect.top + 8, 3, 16)
                    pygame.draw.rect(screen, (80, 80, 80), pole_rect)
                    # Flag triangle
                    flag_points = [
                        (rect.right - 4, rect.top + 8),
                        (rect.right + 6, rect.top + 12),
                        (rect.right - 4, rect.top + 16)
                    ]
                    pygame.draw.polygon(screen, flag_color, flag_points)

    def draw_units(self, screen, units, dt):
        """Draw units with tweened movement, glows, shapes, HP bars, update shake/particles."""
        self._update_shake(dt)
        self._update_particles(dt)
        shake_offset = (self.shake_x, self.shake_y)
        offset = (self.board_offset[0] + shake_offset[0], self.board_offset[1] + shake_offset[1])

        for unit in units:
            if unit.hp <= 0:
                continue

            # Tweened position
            if unit.anim_progress < 1.0:
                progress = self._ease_in_out(unit.anim_progress)
                cur_pos = (
                    unit.pos[0] + (unit.target_pos[0] - unit.pos[0]) * progress,
                    unit.pos[1] + (unit.target_pos[1] - unit.pos[1]) * progress
                )
            else:
                cur_pos = unit.pos

            center = (
                offset[0] + cur_pos[0] * self.tile_size + self.tile_size // 2,
                offset[1] + cur_pos[1] * self.tile_size + self.tile_size // 2
            )

            player_color = self.PLAYER_COLORS[unit.player]

            # Glow effect
            glow_surf = pygame.Surface((44, 44), pygame.SRCALPHA)
            glow_base = tuple(min(255, c + 40) for c in player_color)
            for radius, alpha in [(16, 25), (12, 40), (8, 60)]:
                pygame.draw.circle(glow_surf, (*glow_base, alpha), (22, 22), radius)
            screen.blit(glow_surf, (center[0] - 22, center[1] - 22))

            # Unit body
            if unit.unit_type == 'infantry':
                pygame.draw.circle(screen, player_color, (int(center[0]), int(center[1])), 12)
                # Helmet
                helmet_points = [
                    (center[0] - 8, center[1] - 4),
                    (center[0] + 8, center[1] - 4),
                    (center[0], center[1] + 8)
                ]
                pygame.draw.polygon(screen, (60, 60, 60), helmet_points)
            elif unit.unit_type == 'tank':
                # Tracks
                pygame.draw.rect(screen, (40, 40, 40), (center[0] - 15, center[1] + 4, 30, 8))
                # Hull
                pygame.draw.rect(screen, player_color, (center[0] - 12, center[1] - 8, 24, 14))
                # Turret
                pygame.draw.circle(screen, player_color, (int(center[0] + 10), int(center[1] - 2)), 7)
            elif unit.unit_type == 'artillery':
                # Base
                pygame.draw.rect(screen, player_color, (center[0] - 12, center[1] + 2, 24, 12))
                # Barrel
                end_x = center[0] + 22
                end_y = center[1] - 8
                pygame.draw.line(screen, player_color, center, (end_x, end_y), 5)

            # HP bar
            bar_width = 28
            bar_height = 5
            hp_ratio = unit.hp / unit.max_hp
            bg_rect = (center[0] - bar_width // 2, center[1] - 25, bar_width, bar_height)
            pygame.draw.rect(screen, (40, 40, 40), pygame.Rect(*map(int, bg_rect)))
            fill_width = int(bar_width * hp_ratio)
            fill_color = (0, 200, 0) if hp_ratio > 0.6 else (255, 200, 0) if hp_ratio > 0.3 else (255, 50, 50)
            pygame.draw.rect(screen, fill_color, (bg_rect[0], bg_rect[1], fill_width, bar_height))

        # Draw particles
        for particle in self.particles:
            if particle['life'] > 0:
                ratio = particle['life'] / particle['max_life']
                col = tuple(int(c * ratio) for c in particle['color'])
                size = int(particle['size'] * ratio)
                pos = (int(particle['pos'][0] + self.shake_x), int(particle['pos'][1] + self.shake_y))
                pygame.draw.circle(screen, col, pos, max(1, size))
                pygame.draw.circle(screen, (*col, 100), pos, max(1, size // 2))

    def draw_ui(self, screen, game):
        """Draw UI: sidebar, turn indicator, minimap, day/night tint."""
        # Sidebar background
        sidebar_rect = pygame.Rect(self.window_size[0] - 220, 0, 220, self.window_size[1])
        sidebar_surf = pygame.Surface((220, self.window_size[1]), pygame.SRCALPHA)
        pygame.draw.rect(sidebar_surf, (0, 0, 0, 128), (0, 0, 220, self.window_size[1]))
        screen.blit(sidebar_surf, sidebar_rect.topleft)

        # Turn and funds text
        turn_text = self.font_ui.render(f"{game.current_player.capitalize()}'s Turn {game.turn_count}", True, (255, 255, 255))
        screen.blit(turn_text, (self.window_size[0] - 200, 20))
        funds_text = self.font_small.render(f"Red Funds: {game.funds_red} | Blue: {game.funds_blue}", True, (220, 220, 220))
        screen.blit(funds_text, (self.window_size[0] - 200, 60))

        # Minimap
        minimap_w, minimap_h = 75, 50
        minimap_scale = 5
        minimap_pos = (self.window_size[0] - 80, 100)
        minimap_surf = pygame.Surface((minimap_w, minimap_h), pygame.SRCALPHA)
        for x in range(game.board.width):
            for y in range(game.board.height):
                tile = game.board.get_tile(x, y)
                color = list(self.TERRAIN_COLORS[tile.terrain])
                if tile.owner != 'neutral':
                    pcolor = self.PLAYER_COLORS[tile.owner]
                    color = [int(c * 0.6 + pcolor[i] * 0.4) for i, c in enumerate(color)]
                if tile.hq:
                    color = self.PLAYER_COLORS[tile.owner]
                mx = x * minimap_scale
                my = y * minimap_scale
                pygame.draw.rect(minimap_surf, tuple(color), (mx, my, minimap_scale, minimap_scale))
        screen.blit(minimap_surf, minimap_pos)
        pygame.draw.rect(screen, (255, 255, 255), (*minimap_pos, minimap_w, minimap_h), 2)

        # Day/night cycle tint overlay
        cycle_ratio = 0.7 + 0.3 * math.sin(game.turn_count / 30.0 * math.pi)
        tint_color = (int(60 * cycle_ratio), int(60 * cycle_ratio), int(70 * cycle_ratio))
        tint_surf = pygame.Surface(self.window_size, pygame.SRCALPHA)
        pygame.draw.rect(tint_surf, (*tint_color, 25), (0, 0, *self.window_size))
        screen.blit(tint_surf, (0, 0))