import pygame
import sys
import numpy as np
from collections import deque
from game import BattleshipGame, BattleshipBoard


class PygameDisplay:
    """
    Handles all Pygame rendering for boards, live animation, metrics text/graphs, and event loop.
    Window size 1200x800. Top: boards, bottom: metrics/graphs.
    """

    def __init__(self, screen_width: int = 1200, screen_height: int = 800):
        pygame.init()
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption('Battleship RL Training')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('arial', 24)
        self.small_font = pygame.font.SysFont('arial', 18)
        self.cell_size = 25
        self.margin = 50
        self.colors = {
            'water': (0, 100, 200),
            'ship': (0, 200, 0),
            'hit': (255, 0, 0),
            'miss': (128, 128, 128),
            'grid': (255, 255, 255),
            'bg': (20, 20, 40),
            'text': (255, 255, 255),
            'graph_a': (0, 255, 0),
            'graph_b': (255, 50, 50)
        }
        self.metrics_history_a = deque(maxlen=100)
        self.metrics_history_b = deque(maxlen=100)

    def draw_boards(self, game: BattleshipGame, live: bool = False, delay: float = 0):
        """
        Draw boards: live (perspectives side-by-side) or static preview (own boards).
        Background fill. Delay handled by caller.
        """
        self.screen.fill(self.colors['bg'])

        board_w = 10 * self.cell_size
        board_h = board_w
        own_x = self.margin
        opp_x = self.margin + board_w + 50
        pair_gap_y = board_h + 100

        if live:
            # Agent A perspective (top)
            self._draw_board(game.board_a, True, own_x, self.margin, 'Agent A Own')
            self._draw_board(game.board_b, False, opp_x, self.margin, 'Agent A Opp')
            # Agent B perspective (bottom)
            self._draw_board(game.board_b, True, own_x, self.margin + pair_gap_y, 'Agent B Own')
            self._draw_board(game.board_a, False, opp_x, self.margin + pair_gap_y, 'Agent B Opp')
        else:
            # Static preview: full own boards for A and B
            self._draw_board(game.board_a, True, self.margin, self.margin, 'Agent A')
            self._draw_board(game.board_b, True, self.margin + 600, self.margin, 'Agent B')

    def _draw_board(self, board: BattleshipBoard, is_own_view: bool, x: int, y: int, label: str):
        """Helper: draw a 10x10 board (own or opponent fogged view)."""
        # Label
        text_surf = self.font.render(label, True, self.colors['text'])
        self.screen.blit(text_surf, (x, y - 30))

        # Cells
        for i in range(10):
            for j in range(10):
                color = self._get_cell_color(board, is_own_view, i, j)
                cell_rect = pygame.Rect(
                    x + j * self.cell_size,
                    y + i * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )
                pygame.draw.rect(self.screen, color, cell_rect)
                pygame.draw.rect(self.screen, self.colors['grid'], cell_rect, 1)

    def _get_cell_color(self, board: BattleshipBoard, is_own_view: bool, r: int, c: int) -> tuple:
        """Get color for cell (r,c) based on view type."""
        if is_own_view:
            val = board.grid[r, c]
            if val == 0:
                return self.colors['water']
            elif val == 1:
                return self.colors['ship']
            elif val == 2:
                return self.colors['hit']
        else:
            # Opponent fogged view
            pos = (r, c)
            if pos in board.shots_received:
                if board.grid[r, c] == 2:
                    return self.colors['hit']
                else:
                    return self.colors['miss']
            else:
                return self.colors['water']  # Unknown
        return self.colors['water']  # Fallback

    def draw_metrics(
        self,
        episodes: int,
        wins_a: int,
        wins_b: int,
        draws: int,
        avg_r_a: float,
        avg_r_b: float,
        loss_a: float = 0.0,
        loss_b: float = 0.0
    ):
        """
        Draw metrics text and rolling reward graphs (last 100).
        Append current avgs to history for graphs.
        """
        # Append to histories
        self.metrics_history_a.append(avg_r_a)
        self.metrics_history_b.append(avg_r_b)

        # Text
        metrics_y = 600
        y = metrics_y
        text_surf = self.font.render(f'Episode: {episodes}', True, self.colors['text'])
        self.screen.blit(text_surf, (self.margin, y))
        y += 35

        total = max(1, wins_a + wins_b + draws)
        win_rate_a = (wins_a / total) * 100
        win_rate_b = (wins_b / total) * 100
        draw_rate = (draws / total) * 100
        text_surf = self.font.render(
            f'Win Rates - A: {win_rate_a:.1f}% | B: {win_rate_b:.1f}% | Draw: {draw_rate:.1f}%',
            True, self.colors['text']
        )
        self.screen.blit(text_surf, (self.margin, y))
        y += 35

        text_surf = self.font.render(
            f'Avg Rewards (last 100) - A: {avg_r_a:.2f} | B: {avg_r_b:.2f}',
            True, self.colors['text']
        )
        self.screen.blit(text_surf, (self.margin, y))
        y += 35

        text_surf = self.font.render(
            f'Losses - DQN A: {loss_a:.4f} | PG B: {loss_b:.4f}',
            True, self.colors['text']
        )
        self.screen.blit(text_surf, (self.margin, y))

        # Graphs
        graph_x_start = 450
        graph_y_top = 685
        graph_bottom = 780
        graph_height = graph_bottom - graph_y_top
        graph_x_end = 1150

        def draw_graph(history, color):
            if len(history) < 2:
                return
            min_val = min(history)
            max_val = max(history)
            range_val = max_val - min_val + 1e-6
            points = []
            n = len(history)
            step_x = (graph_x_end - graph_x_start) / max(1, n - 1)
            for i, h in enumerate(history):
                norm = (h - min_val) / range_val
                py = graph_bottom - (norm * graph_height)
                px = graph_x_start + i * step_x
                points.append((int(px), int(py)))
            pygame.draw.lines(self.screen, color, False, points, 3)

        draw_graph(self.metrics_history_a, self.colors['graph_a'])
        draw_graph(self.metrics_history_b, self.colors['graph_b'])

        # Graph label
        label_surf = self.small_font.render(
            'Rolling Avg Episode Rewards (A: green | B: red)',
            True, (200, 200, 200)
        )
        self.screen.blit(label_surf, (graph_x_start, graph_bottom + 5))

    def update(self):
        """Update display, handle events (QUIT/ESC), flip, tick 60 FPS."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                pygame.quit()
                sys.exit()
        pygame.display.flip()
        self.clock.tick(60)