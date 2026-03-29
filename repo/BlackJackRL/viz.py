"""
Pygame Visualizer module for BlackjackBetOptimizer.
Handles real-time rendering of blackjack hands, table, stats, and post-hand visualization.
"""

import pygame
import time
from pygame.locals import *
from typing import List, Tuple
from game import Card, Hand, BasicStrategy, Game, Shoe
from agent import BettingAgent


class Visualizer:
    """
    Handles Pygame rendering of table, cards, stats, and animations for visualization.
    Provides simple static post-hand visualization with auto-advance after 10s or ESC.
    """

    def __init__(self, width: int = 800, height: int = 600):
        """
        Initialize Pygame display, fonts, felt background, and card dimensions.
        
        Args:
            width: Window width in pixels.
            height: Window height in pixels.
        """
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Blackjack Bet Optimizer - RL Visualization")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 32)
        self.small_font = pygame.font.Font(None, 24)
        
        # Green felt background
        self.felt = pygame.Surface((width, height))
        self.felt.fill((20, 120, 20))
        
        # Card dimensions
        self.card_width = 55
        self.card_height = 85
        self.card_spacing = 65

    def draw_card(self, surface: pygame.Surface, card: Card, rect: pygame.Rect) -> None:
        """
        Draw a single card as white rectangle with black border and centered card text.
        
        Args:
            surface: Target pygame surface.
            card: Card object to render.
            rect: Position and size rectangle for card.
        """
        pygame.draw.rect(surface, (255, 255, 255), rect)
        pygame.draw.rect(surface, (0, 0, 0), rect, 3)
        text = self.small_font.render(str(card), True, (0, 0, 0))
        text_rect = text.get_rect(center=rect.center)
        surface.blit(text, text_rect)

    def draw_hand(self, surface: pygame.Surface, hand: Hand, pos: Tuple[int, int], 
                  is_dealer: bool = False) -> None:
        """
        Draw a complete hand of cards starting at pos (full reveal for post-hand viz).
        
        Args:
            surface: Target pygame surface.
            hand: Hand object containing cards.
            pos: (x, y) starting position for first card.
            is_dealer: Dealer flag (unused for drawing, full reveal post-hand).
        """
        x, y = pos
        for i, card in enumerate(hand.cards):
            rect = pygame.Rect(x + i * self.card_spacing, y, 
                               self.card_width, self.card_height)
            self.draw_card(surface, card, rect)

    def draw_text(self, surface: pygame.Surface, text: str, pos: Tuple[int, int], 
                  color: Tuple[int, int, int] = (255, 255, 255)) -> None:
        """
        Draw text on surface using small font at top-left position.
        
        Args:
            surface: Target pygame surface.
            text: Text string to render.
            pos: (x, y) position.
            color: RGB color tuple.
        """
        surf = self.small_font.render(text, True, color)
        surface.blit(surf, pos)

    def run_viz(self, game: Game, agent: BettingAgent, initial_bankroll: float, 
                bankroll_ref: List[float]) -> float:
        """
        Run static post-hand visualization: fresh shoe, bet/play one hand, display results
        for up to 10s or ESC/quit. Updates bankroll_ref[0] in-place.
        
        Args:
            game: Game instance.
            agent: BettingAgent for bet decision.
            initial_bankroll: Initial bankroll for bankroll fraction in state.
            bankroll_ref: Mutable list[0] holding current bankroll.
            
        Returns:
            Payout from this hand (negative for losses).
        """
        game.shoe.shuffle()  # Fresh shoe for viz
        tc = game.shoe.get_true_count()
        bet = agent.get_bet(initial_bankroll, bankroll_ref[0], tc)
        if bet <= 0:
            return 0.0
        
        payout = game.play_hand(bet)
        bankroll_ref[0] += payout
        
        running = True
        start_time = time.time()
        max_viz_time = 10.0
        
        while running:
            for event in pygame.event.get():
                if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                    running = False
            
            self.screen.blit(self.felt, (0, 0))
            
            # Table areas
            pygame.draw.rect(self.screen, (101, 67, 33), (50, 80, 700, 120))   # Dealer
            pygame.draw.rect(self.screen, (101, 67, 33), (50, 420, 700, 120))  # Player
            
            # Hands (full reveal post-hand)
            self.draw_hand(self.screen, game.dealer_hand, (100, 100), is_dealer=True)
            py = 440
            for ph in game.player_hands:
                self.draw_hand(self.screen, ph, (100, py))
                py += 25 if len(game.player_hands) > 1 else 0
            
            # Stats
            self.draw_text(self.screen, f'Bet: ${bet:.0f}', (600, 60), (255, 255, 0))
            self.draw_text(self.screen, f'Payout: ${payout:.0f}', (600, 90), (0, 255, 0))
            self.draw_text(self.screen, f'Bankroll: ${bankroll_ref[0]:,.0f}', (600, 120), (255, 255, 255))
            
            # Counts
            self.draw_text(self.screen, f'True Count: {tc:.1f}', (50, 350), (255, 255, 0))
            self.draw_text(self.screen, f'Running Count: {game.shoe.running_count}', (250, 350), (255, 255, 255))
            
            # Basic strategy approx (first player hand, initial state proxy)
            bs = BasicStrategy()
            if game.player_hands and game.player_hands[0].cards:
                ph = game.player_hands[0]
                dup = game.dealer_hand.cards[0].rank if game.dealer_hand.cards else 2
                if len(ph.cards) == 2 and ph.cards[0].rank == ph.cards[1].rank:
                    action = bs.get_pair_action(ph.cards[0].rank, dup)
                else:
                    action = bs.get_action(ph.total, dup, ph.is_soft)
                self.draw_text(self.screen, f'Basic Strategy: {action}', (50, 380), (0, 255, 0))
            
            pygame.display.flip()
            self.clock.tick(30)
            
            if time.time() - start_time > max_viz_time:
                running = False
        
        return payout