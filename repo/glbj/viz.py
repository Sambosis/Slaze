"""
Pygame visualization module for BlackjackBetOptimizer.
Provides real-time graphical display of gameplay, training progress, HUD stats,
bankroll graph, and animated hand playthroughs.
"""

import pygame
from typing import List, Dict, Tuple, Optional


class Visualizer:
    """
    Handles Pygame rendering of the blackjack game table, cards, HUD stats,
    training statistics, and bankroll line graph in an 800x600 window.
    Supports controls: Space (pause/step), 's' (save), 'l' (load), 'q'/close (quit).
    """

    def __init__(self):
        """Initialize Pygame, screen, fonts, clock, and internal state."""
        pygame.init()
        self.screen: pygame.Surface = pygame.display.set_mode((800, 600))
        pygame.display.set_caption('Blackjack Bet Optimizer')
        self.font: pygame.Font = pygame.font.Font(None, 24)
        self.small_font: pygame.Font = pygame.font.Font(None, 18)
        self.clock: pygame.time.Clock = pygame.time.Clock()
        self.bankroll_history: List[float] = []
        self.paused: bool = False

    def run_event_loop(self) -> Optional[str]:
        """
        Process Pygame events and user input.
        Returns command strings for 'save', 'load', 'quit', or None.
        Toggles pause on Space.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return 'quit'
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_s:
                    return 'save'
                elif event.key == pygame.K_l:
                    return 'load'
                elif event.key == pygame.K_q:
                    return 'quit'
        return None

    def draw_table(self) -> None:
        """Draw the green casino table background and player/dealer areas."""
        self.screen.fill((0, 100, 0))  # Green felt table
        # Player area
        pygame.draw.rect(self.screen, (50, 50, 50), (50, 400, 700, 150))
        # Dealer area
        pygame.draw.rect(self.screen, (50, 50, 50), (50, 100, 700, 150))

    def draw_card(self, card: int, pos: Tuple[int, int], is_back: bool = False) -> None:
        """
        Draw a single card as a white rectangle with rank text.
        If is_back, draw a red back.
        """
        if is_back:
            pygame.draw.rect(self.screen, (255, 0, 0), (pos[0], pos[1], 60, 90))
            return
        rank_str = str(card) if card < 10 else ('10' if card == 10 else 'A')
        color = (255, 255, 255) if card == 11 else (0, 0, 0)
        pygame.draw.rect(self.screen, (255, 255, 255), (pos[0], pos[1], 60, 90))
        pygame.draw.rect(self.screen, (0, 0, 0), (pos[0], pos[1], 60, 90), 2)
        text_surf = self.font.render(rank_str, True, color)
        text_rect = text_surf.get_rect(center=(pos[0] + 30, pos[1] + 45))
        self.screen.blit(text_surf, text_rect)

    def draw_hand(self, cards: List[int], pos: Tuple[int, int], is_dealer: bool = False) -> None:
        """
        Draw a hand of cards fanned out horizontally from pos.
        is_dealer unused (for future hole card back).
        """
        for i, card in enumerate(cards):
            self.draw_card(card, (pos[0] + i * 65, pos[1]))

    def draw_hud(self, bankroll: float, bet: float, running_count: int, true_count: float, stats: Dict) -> None:
        """Draw top-left HUD with bankroll, bet, counts, epsilon, alpha."""
        texts = [
            f'Bankroll: ${bankroll:.0f}',
            f'Bet: ${bet:.0f}',
            f'RC: {running_count}',
            f'TC: {true_count:.1f}',
            f'Epsilon: {stats.get("epsilon", 0):.3f}',
            f'Alpha: {stats.get("alpha", 0):.3f}'
        ]
        for i, text in enumerate(texts):
            surf = self.small_font.render(text, True, (255, 255, 255))
            self.screen.blit(surf, (10, 10 + i * 20))

    def draw_stats(self, stats: Dict) -> None:
        """Draw right-side stats panel: avg return, win ratio, avg bet, max DD."""
        y = 560  # Below player area
        texts = [
            f'Avg Return: {stats.get("avg_return", 0):.4f}',
            f'Win Ratio: {stats.get("win_ratio", 0):.3f}',
            f'Avg Bet: {stats.get("avg_bet", 0):.1f}',
            f'Max DD: -${stats.get("max_drawdown", 0):.0f}'
        ]
        for i, text in enumerate(texts):
            surf = self.font.render(text, True, (255, 255, 0))
            self.screen.blit(surf, (600, y + i * 25))

    def draw_graph(self, history: List[float]) -> None:
        """
        Draw line graph of bankroll_history (last 500 points) in top-left area.
        Scaled to fit 10,300,500x150 rectangle.
        """
        if len(history) < 2:
            return
        left, top, w, h = 10, 300, 500, 150
        points: List[Tuple[int, int]] = []
        min_b = min(history)
        max_b = max(history)
        range_b = max_b - min_b if max_b > min_b else 1.0
        recent = history[-500:]
        n_points = len(recent)
        for i, b in enumerate(recent):
            x = int(left + (i / max(1, n_points - 1)) * w)
            y = int(top + h - ((b - min_b) / range_b) * h)
            points.append((x, y))
        if len(points) > 1:
            pygame.draw.lines(self.screen, (0, 255, 0), False, points, 2)

    def animate_hand(
        self,
        game: 'BlackjackGame',
        bet: float,
        strategy: 'BasicStrategy',
        counter: 'CardCounter',
        player_hands: Optional[List[List[int]]] = None,
        dealer_cards: Optional[List[int]] = None
    ) -> None:
        """
        Animate a full hand playthrough: if hands None, play the hand fully.
        Draws final player hands (up to 3 shown), dealer hand, bet.
        Waits 2 seconds for viewing.
        """
        if player_hands is None or dealer_cards is None:
            _, player_hands, dealer_cards = game.play_hand(bet, strategy, counter)
        self.draw_table()
        # Draw up to 3 player hands
        for i, hand in enumerate(player_hands[:3]):
            self.draw_hand(hand, (100 + i * 200, 420))
        self.draw_hand(dealer_cards, (100, 120))
        bet_surf = self.font.render(f'Bet: ${bet:.0f}', True, (255, 255, 0))
        self.screen.blit(bet_surf, (350, 450))
        pygame.display.flip()
        pygame.time.wait(2000)

    def update(
        self,
        bankroll: float,
        bet: float,
        counter: 'CardCounter',
        stats: Dict,
        history: List[float],
        player_hands: Optional[List[List[int]]] = None,
        dealer_cards: Optional[List[int]] = None,
        game: Optional['BlackjackGame'] = None,
        strategy: Optional['BasicStrategy'] = None
    ) -> None:
        """
        Full screen update: table, HUD, stats, graph, optional hands.
        Computes true count using game shoe if provided.
        """
        self.draw_table()
        decks_rem = len(game.shoe) / (52 * game.num_decks) if game else 1.0
        true_count = counter.get_true_count(decks_rem)
        self.draw_hud(bankroll, bet, counter.running_count, true_count, stats)
        self.draw_stats(stats)
        self.draw_graph(history)
        if player_hands and dealer_cards:
            self.draw_hand(dealer_cards, (100, 120))
            for i, hand in enumerate(player_hands[:3]):
                self.draw_hand(hand, (100 + i * 200, 420))
        pygame.display.flip()
        self.clock.tick(60)