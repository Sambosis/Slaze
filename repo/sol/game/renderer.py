"""
renderer.py - PyGame rendering implementation for the solitaire game.

This module handles all visualization aspects of the solitaire game, including
drawing cards, rendering the game table, displaying scores, and animating card
movements. It uses configuration from config.py for dimensions and colors.
"""

import pygame
import os
from typing import Tuple, List, Optional
from game.solitaire import SolitaireGame, Card
import config

class GameRenderer:
    """
    Handles all PyGame rendering operations for the solitaire game.

    This class is responsible for visually representing the game state, including
    cards, piles, scores, and game status. It also handles animations and
    highlighting of valid moves.

    Attributes:
        screen (pygame.Surface): The main display surface.
        clock (pygame.time.Clock): Controls the frame rate.
        font (pygame.font.Font): Font for rendering text.
        card_images (dict): Cache for card images to improve performance.
        background (pygame.Surface): The game background surface.
    """

    def __init__(self, screen: pygame.Surface):
        """
        Initialize the GameRenderer with the provided screen surface.

        Args:
            screen: The PyGame display surface to render on.
        """
        self.screen = screen
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(config.FONT_NAME, config.FONT_SIZE)
        self.large_font = pygame.font.SysFont(config.FONT_NAME, config.FONT_SIZE_LARGE)
        self.card_images = {}
        self.background = None
        self._load_assets()

    def _load_assets(self) -> None:
        """
        Load game assets including card images and background.
        """
        # Load background image or create a solid color background
        if os.path.exists(config.BACKGROUND_IMAGE_PATH):
            try:
                self.background = pygame.image.load(config.BACKGROUND_IMAGE_PATH)
                self.background = pygame.transform.scale(self.background,
                                                       (config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
            except pygame.error:
                self.background = None
        else:
            self.background = pygame.Surface((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
            self.background.fill(config.GAME_COLORS['TABLE'])

        # Load card images if available, otherwise use programmatic rendering
        self._load_card_images()

    def _load_card_images(self) -> None:
        """
        Load card images from the assets directory or initialize for programmatic rendering.
        """
        # Check if card images are available
        if os.path.exists(config.CARDS_IMAGE_DIR):
            try:
                # Load card back image
                back_path = os.path.join(config.CARDS_IMAGE_DIR, 'card_back.png')
                if os.path.exists(back_path):
                    self.card_back = pygame.image.load(back_path)
                    self.card_back = pygame.transform.scale(self.card_back,
                                                          (config.CARD_WIDTH, config.CARD_HEIGHT))
                else:
                    self.card_back = None
            except pygame.error:
                self.card_back = None
        else:
            self.card_back = None

        # Initialize card face images (will be created programmatically if not found)
        self.card_images = {}

    def _create_card_surface(self, card: Card) -> pygame.Surface:
        """
        Create a surface for a card, either using pre-loaded images or programmatic rendering.

        Args:
            card: The card to create a surface for.

        Returns:
            A pygame.Surface representing the card.
        """
        # Check if we have a cached image for this card
        cache_key = f"{card.suit}_{card.rank}_{'face_up' if card.face_up else 'face_down'}"
        if cache_key in self.card_images:
            return self.card_images[cache_key]

        # Create a new surface
        card_surface = pygame.Surface((config.CARD_WIDTH, config.CARD_HEIGHT))

        if not card.face_up:
            # Draw card back
            if self.card_back:
                card_surface.blit(self.card_back, (0, 0))
            else:
                card_surface.fill(config.GAME_COLORS['CARD_BACK'])
                # Draw a simple pattern for the card back
                pygame.draw.rect(card_surface, (0, 150, 0),
                                (5, 5, config.CARD_WIDTH - 10, config.CARD_HEIGHT - 10), 2)
                pygame.draw.line(card_surface, (0, 200, 0),
                                (10, 10), (config.CARD_WIDTH - 10, config.CARD_HEIGHT - 10), 2)
                pygame.draw.line(card_surface, (0, 200, 0),
                                (config.CARD_WIDTH - 10, 10), (10, config.CARD_HEIGHT - 10), 2)
        else:
            # Draw card face
            card_surface.fill(config.GAME_COLORS['CARD_FACE'])
            pygame.draw.rect(card_surface, (0, 0, 0),
                           (0, 0, config.CARD_WIDTH, config.CARD_HEIGHT), 2)

            # Draw suit symbol and rank
            suit_color = (255, 0, 0) if card.is_red() else (0, 0, 0)

            # Top-left corner
            self._draw_suit(card_surface, card.suit, 10, 10, suit_color)
            rank_text = self.font.render(card.rank, True, suit_color)
            card_surface.blit(rank_text, (10, 30))

            # Bottom-right corner (upside down)
            self._draw_suit(card_surface, card.suit, config.CARD_WIDTH - 30, config.CARD_HEIGHT - 30,
                           suit_color, upside_down=True)
            rank_text = self.font.render(card.rank, True, suit_color)
            rank_text = pygame.transform.rotate(rank_text, 180)
            card_surface.blit(rank_text, (config.CARD_WIDTH - 30, config.CARD_HEIGHT - 50))

            # Center suit symbol
            self._draw_suit(card_surface, card.suit, config.CARD_WIDTH // 2 - 15,
                           config.CARD_HEIGHT // 2 - 15, suit_color, large=True)

        # Cache the surface
        self.card_images[cache_key] = card_surface
        return card_surface

    def _draw_suit(self, surface: pygame.Surface, suit: str, x: int, y: int,
                  color: Tuple[int, int, int], upside_down: bool = False, large: bool = False) -> None:
        """
        Draw a suit symbol on a surface.

        Args:
            surface: The surface to draw on.
            suit: The suit to draw (Hearts, Diamonds, Clubs, Spades).
            x: The x-coordinate.
            y: The y-coordinate.
            color: The color to draw the suit in.
            upside_down: Whether to draw the suit upside down.
            large: Whether to draw a large version of the suit.
        """
        size = 30 if large else 15
        half_size = size // 2

        if upside_down:
            if suit == 'Hearts':
                # Draw upside-down heart
                pygame.draw.circle(surface, color, (x + half_size, y + half_size), half_size)
                pygame.draw.circle(surface, color, (x + half_size * 3, y + half_size), half_size)
                pygame.draw.polygon(surface, color, [
                    (x, y + half_size),
                    (x + half_size * 2, y + size * 2),
                    (x + half_size * 4, y + half_size)
                ])
            elif suit == 'Diamonds':
                # Draw upside-down diamond
                pygame.draw.polygon(surface, color, [
                    (x + half_size * 2, y),
                    (x + half_size * 4, y + half_size * 2),
                    (x + half_size * 2, y + half_size * 4),
                    (x, y + half_size * 2)
                ])
            elif suit == 'Clubs':
                # Draw upside-down club
                pygame.draw.circle(surface, color, (x + half_size, y + half_size), half_size)
                pygame.draw.circle(surface, color, (x + half_size * 3, y + half_size), half_size)
                pygame.draw.circle(surface, color, (x + half_size * 2, y + half_size * 3), half_size)
                pygame.draw.rect(surface, color,
                                (x + half_size, y + half_size * 2, half_size * 2, half_size))
            elif suit == 'Spades':
                # Draw upside-down spade
                pygame.draw.circle(surface, color, (x + half_size, y + half_size), half_size)
                pygame.draw.circle(surface, color, (x + half_size * 3, y + half_size), half_size)
                pygame.draw.polygon(surface, color, [
                    (x, y + half_size * 2),
                    (x + half_size * 2, y + half_size * 4),
                    (x + half_size * 4, y + half_size * 2)
                ])
        else:
            if suit == 'Hearts':
                # Draw heart
                pygame.draw.circle(surface, color, (x + half_size, y + half_size), half_size)
                pygame.draw.circle(surface, color, (x + half_size * 3, y + half_size), half_size)
                pygame.draw.polygon(surface, color, [
                    (x + half_size * 2, y + size * 2),
                    (x, y + half_size),
                    (x + half_size * 4, y + half_size)
                ])
            elif suit == 'Diamonds':
                # Draw diamond
                pygame.draw.polygon(surface, color, [
                    (x + half_size * 2, y),
                    (x + half_size * 4, y + half_size * 2),
                    (x + half_size * 2, y + half_size * 4),
                    (x, y + half_size * 2)
                ])
            elif suit == 'Clubs':
                # Draw club
                pygame.draw.circle(surface, color, (x + half_size, y + half_size), half_size)
                pygame.draw.circle(surface, color, (x + half_size * 3, y + half_size), half_size)
                pygame.draw.circle(surface, color, (x + half_size * 2, y + half_size * 3), half_size)
                pygame.draw.rect(surface, color,
                                (x + half_size, y + half_size * 2, half_size * 2, half_size))
            elif suit == 'Spades':
                # Draw spade
                pygame.draw.circle(surface, color, (x + half_size, y + half_size), half_size)
                pygame.draw.circle(surface, color, (x + half_size * 3, y + half_size), half_size)
                pygame.draw.polygon(surface, color, [
                    (x + half_size * 2, y),
                    (x, y + half_size * 2),
                    (x + half_size * 4, y + half_size * 2)
                ])

    def draw_card(self, card: Card, position: Tuple[int, int]) -> None:
        """
        Draw a single card at the specified position.

        Args:
            card: The card to draw.
            position: The (x, y) position to draw the card at.
        """
        card_surface = self._create_card_surface(card)
        self.screen.blit(card_surface, position)

    def render(self, game: SolitaireGame, highlight_valid_moves: bool = False) -> None:
        """
        Render the entire game state to the screen.

        Args:
            game: The SolitaireGame instance to render.
            highlight_valid_moves: Whether to highlight valid moves.
        """
        # Clear the screen with the background
        self.screen.blit(self.background, (0, 0))

        # Draw the title
        title_text = self.large_font.render("Solitaire with RL", True, config.GAME_COLORS['TEXT'])
        self.screen.blit(title_text, (config.SCREEN_WIDTH // 2 - title_text.get_width() // 2, 10))

        # Draw score
        score_text = self.font.render(f"Score: {game.score}", True, config.GAME_COLORS['TEXT'])
        self.screen.blit(score_text, (10, 10))

        # Draw stock and waste piles
        self._draw_stock_pile(game)
        self._draw_waste_pile(game)

        # Draw foundation piles
        self._draw_foundation_piles(game)

        # Draw tableau piles
        self._draw_tableau_piles(game, highlight_valid_moves)

        # Update the display
        pygame.display.flip()

    def _draw_stock_pile(self, game: SolitaireGame) -> None:
        """
        Draw the stock pile.

        Args:
            game: The SolitaireGame instance.
        """
        stock_pos = config.STOCK_POS
        if not game.stock.is_empty():
            # Draw the top card of the stock (face down)
            card_surface = self._create_card_surface(Card('Hearts', 'Ace', face_up=False))
            self.screen.blit(card_surface, stock_pos)

            # Draw the number of cards remaining in stock
            count_text = self.font.render(str(len(game.stock)), True, config.GAME_COLORS['TEXT'])
            self.screen.blit(count_text, (stock_pos[0] + 5, stock_pos[1] + 5))

    def _draw_waste_pile(self, game: SolitaireGame) -> None:
        """
        Draw the waste pile.

        Args:
            game: The SolitaireGame instance.
        """
        waste_pos = config.WASTE_POS
        if len(game.waste) > 0:
            # Draw the top 3 cards of the waste pile (or fewer if less than 3)
            num_to_show = min(3, len(game.waste))
            for i in range(num_to_show):
                card = game.waste[-num_to_show + i]
                card_pos = (waste_pos[0] + i * config.PILE_SPACING, waste_pos[1])
                self.draw_card(card, card_pos)

    def _draw_foundation_piles(self, game: SolitaireGame) -> None:
        """
        Draw the foundation piles.

        Args:
            game: The SolitaireGame instance.
        """
        for i in range(4):
            foundation = game.foundations[i]
            if len(foundation) > 0:
                # Draw the top card of the foundation
                top_card = foundation[-1]
                card_pos = (config.FOUNDATIONS_START_X + i * (config.CARD_WIDTH + config.PILE_SPACING),
                           config.TOP_AREA_Y)
                self.draw_card(top_card, card_pos)

    def _draw_tableau_piles(self, game: SolitaireGame, highlight_valid_moves: bool = False) -> None:
        """
        Draw the tableau piles.

        Args:
            game: The SolitaireGame instance.
            highlight_valid_moves: Whether to highlight valid moves.
        """
        for i in range(7):
            pile = game.tableau[i]
            if len(pile) > 0:
                # Draw each card in the pile
                for j, card in enumerate(pile):
                    # Calculate position with offset for stacked cards
                    card_pos = (
                        config.TABLEAU_START_X + i * config.TABLEAU_SPACING,
                        config.TABLEAU_START_Y + j * config.TABLEAU_OFFSET
                    )

                    # Highlight if this is a valid move source and highlighting is enabled
                    if highlight_valid_moves and card.face_up:
                        # Check if this card is part of any valid move
                        valid_moves = game.get_valid_moves()
                        for move in valid_moves:
                            if move[0] == 'to_foundation' and move[1] == card:
                                # Draw highlight rectangle
                                pygame.draw.rect(self.screen, config.GAME_COLORS['HIGHLIGHT'],
                                                (card_pos[0] - 2, card_pos[1] - 2,
                                                 config.CARD_WIDTH + 4, config.CARD_HEIGHT + 4), 2)
                                break
                            elif move[0] == 'to_tableau' and card in move[1]:
                                # For tableau moves, highlight the bottom card of the sequence
                                if card == move[1][0]:
                                    pygame.draw.rect(self.screen, config.GAME_COLORS['HIGHLIGHT'],
                                                    (card_pos[0] - 2, card_pos[1] - 2,
                                                     config.CARD_WIDTH + 4, config.CARD_HEIGHT + 4), 2)
                                break

                    self.draw_card(card, card_pos)

    def animate_card_movement(self, card: Card, start_pos: Tuple[int, int],
                             end_pos: Tuple[int, int], duration: float = 0.3) -> None:
        """
        Animate a card moving from one position to another.

        Args:
            card: The card to animate.
            start_pos: The starting position.
            end_pos: The ending position.
            duration: The duration of the animation in seconds.
        """
        # Calculate the number of frames based on duration and frame rate
        frames = int(duration * 60)  # Assuming 60 FPS
        if frames <= 0:
            frames = 1

        # Calculate the step size for x and y
        x_step = (end_pos[0] - start_pos[0]) / frames
        y_step = (end_pos[1] - start_pos[1]) / frames

        # Perform the animation
        for i in range(frames):
            # Calculate current position
            current_x = start_pos[0] + x_step * i
            current_y = start_pos[1] + y_step * i

            # Clear the previous position (simple approach - in a real game you'd need better handling)
            # For now, we'll just redraw the entire game state

            # Draw the game state
            self.render(SolitaireGame())  # This is a placeholder - in practice you'd pass the actual game

            # Draw the card at the current position
            self.draw_card(card, (int(current_x), int(current_y)))

            # Update the display
            pygame.display.flip()

            # Control the frame rate
            self.clock.tick(60)

    def render_game_over(self, game: SolitaireGame, won: bool) -> None:
        """
        Render the game over screen.

        Args:
            game: The SolitaireGame instance.
            won: Whether the player won the game.
        """
        # Draw semi-transparent overlay
        overlay = pygame.Surface((config.SCREEN_WIDTH, config.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))

        # Draw game over message
        if won:
            message = "Congratulations! You Won!"
            color = (0, 255, 0)
        else:
            message = "Game Over"
            color = (255, 0, 0)

        message_text = self.large_font.render(message, True, color)
        self.screen.blit(message_text,
                        (config.SCREEN_WIDTH // 2 - message_text.get_width() // 2,
                         config.SCREEN_HEIGHT // 2 - message_text.get_height() // 2))

        # Draw final score
        score_text = self.font.render(f"Final Score: {game.score}", True, config.GAME_COLORS['TEXT'])
        self.screen.blit(score_text,
                        (config.SCREEN_WIDTH // 2 - score_text.get_width() // 2,
                         config.SCREEN_HEIGHT // 2 + 50))

        # Draw restart instruction
        restart_text = self.font.render("Press R to restart", True, config.GAME_COLORS['TEXT'])
        self.screen.blit(restart_text,
                        (config.SCREEN_WIDTH // 2 - restart_text.get_width() // 2,
                         config.SCREEN_HEIGHT // 2 + 100))

        pygame.display.flip()

    def render_training_info(self, episode: int, epsilon: float, average_reward: float) -> None:
        """
        Render training information during RL training.

        Args:
            episode: The current episode number.
            epsilon: The current epsilon value for exploration.
            average_reward: The average reward over recent episodes.
        """
        # Draw training info in the top-right corner
        info_lines = [
            f"Episode: {episode}",
            f"Epsilon: {epsilon:.4f}",
            f"Avg Reward: {average_reward:.2f}"
        ]

        for i, line in enumerate(info_lines):
            text = self.font.render(line, True, config.GAME_COLORS['TEXT'])
            self.screen.blit(text, (config.SCREEN_WIDTH - text.get_width() - 10, 10 + i * 30))