import pygame
import numpy as np
from game import GameState, Unit
from typing import Optional, Tuple

class GameRenderer:
    """
    Pygame-based renderer for the WarGame.
    Handles drawing the board, units with HP bars, fog overlay, bases, UI stats, minimap,
    and smooth unit move animations via interpolation over 30 frames (~0.5s at 60 FPS).
    """
    def __init__(self, screen, clock):
        self.screen = screen
        self.clock = clock
        self.tile_size = 40
        self.board_width = 15 * self.tile_size
        self.board_height = 15 * self.tile_size
        self.colors = {
            0: (0, 255, 0),        # grass
            1: (0, 100, 0),        # forest
            2: (139, 69, 19),      # hill
            'red_unit': (255, 0, 0),
            'blue_unit': (0, 0, 255),
            'base_red': (200, 0, 0),
            'base_blue': (0, 0, 200),
            'fog': (0, 0, 0, 128),
            'hp_green': (0, 255, 0),
            'hp_red': (255, 0, 0),
            'white': (255, 255, 255),
            'black': (0, 0, 0),
            'gray': (100, 100, 100),
        }
        self.font = pygame.font.SysFont('arial', 20)
        self.small_font = pygame.font.SysFont('arial', 14)
        self.mini_size = 150
        self.mini_tile = 10  # 150 / 15
        self.animations = {}  # key: dict[start_pos, target_pos, progress, unit]

    def draw(self, state: GameState, agent: str = 'spectate') -> None:
        """
        Draw the full game state: terrain, fog (if not spectate), animated units,
        static units, bases, minimap, UI.
        Updates animations progress (assumes ~60 FPS calls).
        """
        self.screen.fill(self.colors['black'])

        # Draw terrain
        for r in range(state.height):
            for c in range(state.width):
                terr_id = state.terrain[r, c]
                color = self.colors[terr_id]
                rect = pygame.Rect(
                    c * self.tile_size,
                    r * self.tile_size,
                    self.tile_size,
                    self.tile_size
                )
                pygame.draw.rect(self.screen, color, rect)

        # Fog overlay (if not spectating)
        if agent != 'spectate':
            fog = state.fog_red if agent == 'red' else state.fog_blue
            fog_surf = pygame.Surface((self.board_width, self.board_height), pygame.SRCALPHA)
            fog_surf.fill((0, 0, 0, 0))
            for r in range(state.height):
                for c in range(state.width):
                    if not fog[r, c]:
                        rect = pygame.Rect(
                            c * self.tile_size,
                            r * self.tile_size,
                            self.tile_size,
                            self.tile_size
                        )
                        pygame.draw.rect(fog_surf, self.colors['fog'], rect)
            self.screen.blit(fog_surf, (0, 0))

        # Draw bases (under fog ok, visible if lit)
        # Red base at (0,0)
        base_color = self.colors['base_red'] if state.bases_red_alive else (100, 0, 0)
        pygame.draw.rect(self.screen, base_color, (2, 2, self.tile_size - 4, self.tile_size - 4))
        # Blue base at (14,14)
        base_color = self.colors['base_blue'] if state.bases_blue_alive else (0, 100, 100)
        bx = 14 * self.tile_size + 2
        by = 14 * self.tile_size + 2
        pygame.draw.rect(self.screen, base_color, (bx, by, self.tile_size - 4, self.tile_size - 4))

        # Update and draw animations (lerped units)
        to_remove = []
        for key, anim in self.animations.items():
            anim['progress'] += 1.0 / 30.0
            if anim['progress'] >= 1.0:
                anim['unit'].pos = anim['target_pos']
                to_remove.append(key)
            else:
                dx = anim['target_pos'][0] - anim['start_pos'][0]
                dy = anim['target_pos'][1] - anim['start_pos'][1]
                lerped_pos = (
                    anim['start_pos'][0] + anim['progress'] * dx,
                    anim['start_pos'][1] + anim['progress'] * dy
                )
                self._draw_unit(anim['unit'], lerped_pos)
        for key in to_remove:
            del self.animations[key]

        # Draw static units (skip animating ones)
        show_all = (agent == 'spectate')
        if show_all:
            all_units = state.units_red + state.units_blue
        else:
            own_units = state.units_red if agent == 'red' else state.units_blue
            enemy_units = state.units_blue if agent == 'red' else state.units_red
            fog = state.fog_red if agent == 'red' else state.fog_blue
            visible_units = []
            for unit in own_units:
                if unit.hp > 0 and fog[unit.pos[0], unit.pos[1]]:
                    visible_units.append(unit)
            for unit in enemy_units:
                if unit.hp > 0 and fog[unit.pos[0], unit.pos[1]]:
                    visible_units.append(unit)
            all_units = visible_units

        for unit in all_units:
            key = f"{unit.owner}_{unit.slot_id}"
            if key in self.animations:
                continue  # Already drawn as animated
            self._draw_unit(unit)

        # Minimap (bottom-right of board area)
        self._draw_minimap(state)

        # UI (right panel)
        self._draw_ui(state)

    def _draw_unit(self, unit: Unit, pos: Optional[Tuple[int, int]] = None) -> None:
        """Draw unit circle, type icon, HP bar at given pos (or unit.pos)."""
        if pos is None:
            pos = unit.pos
        cx = int(pos[1] * self.tile_size + self.tile_size / 2)
        cy = int(pos[0] * self.tile_size + self.tile_size / 2)
        color = self.colors['red_unit'] if unit.owner == 'red' else self.colors['blue_unit']
        pygame.draw.circle(self.screen, color, (cx, cy), 15)

        # Type icon (I/T)
        icon = unit.type[0].upper()
        text_surf = self.small_font.render(icon, True, self.colors['white'])
        text_rect = text_surf.get_rect(center=(cx, cy))
        self.screen.blit(text_surf, text_rect)

        # HP bar above
        max_bar_w = 20
        bar_ratio = max(0.0, unit.hp / unit.max_hp)
        bar_w = int(bar_ratio * max_bar_w)
        bar_h = 4
        bar_x = cx - max_bar_w // 2
        bar_y = cy - 25
        # BG
        pygame.draw.rect(self.screen, (50, 50, 50), (bar_x, bar_y, max_bar_w, bar_h))
        # Fill
        if bar_w > 0:
            bar_color = self.colors['hp_green'] if unit.hp > 0.5 * unit.max_hp else self.colors['hp_red']
            pygame.draw.rect(self.screen, bar_color, (bar_x, bar_y, bar_w, bar_h))
        # Border
        pygame.draw.rect(self.screen, self.colors['white'], (bar_x, bar_y, max_bar_w, bar_h), 1)

    def _draw_minimap(self, state: GameState) -> None:
        """Draw 150x150 minimap (terrain, units dots, bases) at board bottom-right."""
        mini_x = self.board_width + 20
        mini_y = self.board_height - self.mini_size - 20
        # BG
        pygame.draw.rect(self.screen, self.colors['gray'], (mini_x, mini_y, self.mini_size, self.mini_size))
        # Terrain
        for r in range(state.height):
            for c in range(state.width):
                terr_id = state.terrain[r, c]
                color = self.colors[terr_id][:3]  # No alpha
                rect = pygame.Rect(
                    mini_x + c * self.mini_tile,
                    mini_y + r * self.mini_tile,
                    self.mini_tile,
                    self.mini_tile
                )
                pygame.draw.rect(self.screen, color, rect)
        # Red units
        for unit in state.units_red:
            if unit.hp > 0:
                ux = mini_x + unit.pos[1] * self.mini_tile + self.mini_tile // 2
                uy = mini_y + unit.pos[0] * self.mini_tile + self.mini_tile // 2
                pygame.draw.circle(self.screen, self.colors['red_unit'][:3], (int(ux), int(uy)), 3)
        # Blue units
        for unit in state.units_blue:
            if unit.hp > 0:
                ux = mini_x + unit.pos[1] * self.mini_tile + self.mini_tile // 2
                uy = mini_y + unit.pos[0] * self.mini_tile + self.mini_tile // 2
                pygame.draw.circle(self.screen, self.colors['blue_unit'][:3], (int(ux), int(uy)), 3)
        # Bases
        if state.bases_red_alive:
            pygame.draw.rect(self.screen, self.colors['base_red'][:3],
                             (mini_x + 2, mini_y + 2, 6, 6))
        if state.bases_blue_alive:
            bx = mini_x + 14 * self.mini_tile + 1
            by = mini_y + 14 * self.mini_tile + 1
            pygame.draw.rect(self.screen, self.colors['base_blue'][:3], (bx, by, 6, 6))
        # Label
        label = self.small_font.render("Minimap", True, self.colors['white'])
        self.screen.blit(label, (mini_x, mini_y - 20))

    def _draw_ui(self, state: GameState) -> None:
        """Draw right-panel UI: resources, turn, base status, training placeholders."""
        ui_x = self.board_width + 20
        ui_y = 20
        texts = [
            f"Red Resources: {state.resources_red}",
            f"Blue Resources: {state.resources_blue}",
            f"Turn: {state.turn_count}",
            f"Red Base HP: {state.base_hp_red} ({'Alive' if state.bases_red_alive else 'Dead'})",
            f"Blue Base HP: {state.base_hp_blue} ({'Alive' if state.bases_blue_alive else 'Dead'})",
            "Winrate: --%",
            "Avg Reward: --",
            "Total Episodes: --",
        ]
        for text_str in texts:
            surf = self.font.render(text_str, True, self.colors['white'])
            self.screen.blit(surf, (ui_x, ui_y))
            ui_y += 30

    def animate_move(self, unit: Unit, target_pos: Tuple[int, int]) -> None:
        """
        Start smooth animation for unit from current pos to target_pos.
        Animation handled automatically in draw() over ~30 frames (0.5s @60FPS).
        Call before updating unit.pos (e.g., with intended pos from action).
        """
        if unit.pos == target_pos:
            return
        key = f"{unit.owner}_{unit.slot_id}"
        self.animations[key] = {
            'unit': unit,
            'start_pos': unit.pos,
            'target_pos': target_pos,
            'progress': 0.0
        }