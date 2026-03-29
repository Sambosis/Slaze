"""
Pygame-based rendering functionality for the Blackjack Reinforcement Learning Agent.

This module provides comprehensive visualization of the blackjack game state,
statistics, and training progress using Pygame. It handles card display, UI elements,
text rendering, and game statistics presentation for both training and gameplay modes.
"""

import pygame
import os
from typing import Dict, List, Tuple, Optional
import config

# Initialize pygame fonts
pygame.font.init()

class CardRenderer:
    """
    Handles rendering of individual playing cards.

    This class manages loading card images, scaling them appropriately,
    and rendering them to the screen.
    """

    def __init__(self, card_width: int = None, card_height: int = None):
        """
        Initialize the card renderer with specified dimensions.

        Args:
            card_width: Width of card images in pixels (uses config if None)
            card_height: Height of card images in pixels (uses config if None)
        """
        # Use configuration defaults if not specified
        config_vis = config.VISUALIZATION_CONFIG
        self.card_width = card_width or config_vis["card_width"]
        self.card_height = card_height or config_vis["card_height"]
        self.card_images = {}
        self.card_back_image = None
        self._load_card_images()

    def _load_card_images(self) -> None:
        """
        Load card images from the assets directory.

        Card images are expected to follow the naming convention:
        '{rank}_of_{suit}.png' (e.g., 'A_of_Hearts.png')

        If loading fails, generates placeholder cards with basic graphics.
        """
        card_dir = config.PATHS["card_images"]

        try:
            # Load card back image
            back_path = os.path.join(card_dir, "card_back.png")
            if os.path.exists(back_path):
                self.card_back_image = pygame.image.load(back_path)
                self.card_back_image = pygame.transform.scale(
                    self.card_back_image, (self.card_width, self.card_height)
                )

            # Load all card images
            ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
            suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']

            for rank in ranks:
                for suit in suits:
                    filename = f"{rank}_of_{suit}.png"
                    filepath = os.path.join(card_dir, filename)
                    try:
                        if os.path.exists(filepath):
                            img = pygame.image.load(filepath)
                            img = pygame.transform.scale(img, (self.card_width, self.card_height))
                            self.card_images[f"{rank}_{suit}"] = img
                    except pygame.error as e:
                        print(f"Warning: Could not load {filename}: {e}")

        except Exception as e:
            print(f"Error loading card images: {e}")
            # Create placeholder card images if loading fails
            self._create_placeholder_images()

    def _create_placeholder_images(self) -> None:
        """
        Create placeholder card images with basic graphics when loading fails.

        Generates simple card representations using pygame drawing functions.
        """
        # Create a simple colored rectangle for card back
        self.card_back_image = pygame.Surface((self.card_width, self.card_height))
        self.card_back_image.fill((0, 0, 128))  # Blue card back
        pygame.draw.rect(self.card_back_image, (255, 255, 255),
                        (5, 5, self.card_width-10, self.card_height-10), 2)

        # Create placeholder for each card
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']

        for rank in ranks:
            for suit in suits:
                card_surface = pygame.Surface((self.card_width, self.card_height))
                card_surface.fill((255, 255, 255))  # White background

                # Draw border
                pygame.draw.rect(card_surface, (0, 0, 0),
                               (0, 0, self.card_width, self.card_height), 3)

                # Determine suit color
                suit_color = (255, 0, 0) if suit in ['Hearts', 'Diamonds'] else (0, 0, 0)

                # Draw rank and suit
                font = pygame.font.Font(None, 32)
                rank_text = font.render(rank, True, suit_color)
                suit_symbol = {'Hearts': '♥', 'Diamonds': '♦', 'Clubs': '♣', 'Spades': '♠'}[suit]
                suit_text = font.render(suit_symbol, True, suit_color)

                # Position text
                card_surface.blit(rank_text, (10, 10))
                card_surface.blit(suit_text, (10, 45))
                # Also show in bottom right (rotated)
                card_surface.blit(pygame.transform.rotate(rank_text, 180),
                                (self.card_width-42, self.card_height-42))
                card_surface.blit(pygame.transform.rotate(suit_text, 180),
                                (self.card_width-42, self.card_height-77))

                self.card_images[f"{rank}_{suit}"] = card_surface

    def get_card_image(self, rank: str, suit: str) -> Optional[pygame.Surface]:
        """
        Get the image for a specific card.

        Args:
            rank: Card rank (2-10, J, Q, K, A)
            suit: Card suit (Hearts, Diamonds, Clubs, Spades)

        Returns:
            Pygame surface with the card image, or None if not found
        """
        key = f"{rank}_{suit}"
        return self.card_images.get(key, None)

    def render_card(self, screen: pygame.Surface, card: Dict, x: int, y: int,
                   face_up: bool = True) -> None:
        """
        Render a card to the screen at the specified position.

        Args:
            screen: Pygame surface to render to
            card: Dictionary containing card information with 'rank' and 'suit' keys
            x: X coordinate to render at
            y: Y coordinate to render at
            face_up: Whether to show the card face up or face down
        """
        if not face_up:
            if self.card_back_image:
                screen.blit(self.card_back_image, (x, y))
            return

        if 'rank' in card and 'suit' in card:
            card_image = self.get_card_image(card['rank'], card['suit'])
            if card_image:
                screen.blit(card_image, (x, y))

class TextRenderer:
    """
    Handles rendering of text elements with various styles and formatting.

    This class provides flexible text rendering with multiple fonts,
    colors, and alignment options for displaying game information.

    Attributes:
        fonts (dict): Dictionary of loaded pygame fonts
        colors (dict): Color configuration from visualization config
    """

    def __init__(self):
        """Initialize the text renderer with default font settings."""
        # Load fonts with configured sizes
        font_sizes = config.VISUALIZATION_CONFIG["font_sizes"]
        self.fonts = {
            'title': pygame.font.Font(None, font_sizes["title"]),
            'stats': pygame.font.Font(None, font_sizes["stats"]),
            'ui': pygame.font.Font(None, font_sizes["ui"]),
            'small': pygame.font.Font(None, 14)
        }

        # Make title font bold
        self.fonts['title'].set_bold(True)

        # Load colors from configuration
        self.colors = config.VISUALIZATION_CONFIG["colors"]

    def render_text(self, screen: pygame.Surface, text: str, x: int, y: int,
                   font_type: str = 'ui', color: Tuple[int, int, int] = None,
                   background: Tuple[int, int, int] = None) -> None:
        """
        Render text to the screen.

        Args:
            screen: Pygame surface to render to
            text: Text to render
            x: X coordinate
            y: Y coordinate
            font_type: Type of font to use ('title', 'stats', 'ui', 'small')
            color: Text color (RGB tuple), uses default if None
            background: Background color (RGB tuple), None for transparent
        """
        if color is None:
            color = self.colors["text"]

        font = self.fonts.get(font_type, self.fonts['ui'])
        if background:
            text_surface = font.render(text, True, color, background)
        else:
            text_surface = font.render(text, True, color)
        screen.blit(text_surface, (x, y))

    def render_centered_text(self, screen: pygame.Surface, text: str, center_x: int, y: int,
                           font_type: str = 'ui', color: Tuple[int, int, int] = None) -> None:
        """
        Render centered text to the screen.

        Args:
            screen: Pygame surface to render to
            text: Text to render
            center_x: Center X coordinate
            y: Y coordinate
            font_type: Type of font to use
            color: Text color (RGB tuple)
        """
        if color is None:
            color = self.colors["text"]

        font = self.fonts.get(font_type, self.fonts['ui'])
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect(centerx=center_x, y=y)
        screen.blit(text_surface, text_rect)

    def render_info_box(self, screen: pygame.Surface, title: str, items: Dict[str, str],
                       x: int, y: int, width: int, height: int) -> None:
        """
        Render an information box with a title and key-value pairs.

        Args:
            screen: Pygame surface to render to
            title: Title of the info box
            items: Dictionary of key-value pairs to display
            x: X coordinate of top-left corner
            y: Y coordinate of top-left corner
            width: Width of the box
            height: Height of the box
        """
        # Create semi-transparent background surface
        box_surface = pygame.Surface((width, height), pygame.SRCALPHA)
        bg_color = (*self.colors["info_box"][:3], 180) if len(self.colors["info_box"]) == 4 else (*self.colors["info_box"], 180)
        box_surface.fill(bg_color)

        screen.blit(box_surface, (x, y))

        # Draw border
        pygame.draw.rect(screen, self.colors["text"], (x, y, width, height), 2)

        # Render title
        self.render_centered_text(screen, title, x + width // 2, y + 10, 'ui')

        # Render items
        line_height = 25
        for i, (key, value) in enumerate(items.items()):
            key_text = f"{key}:"
            value_text = str(value)

            self.render_text(screen, key_text, x + 10, y + 40 + i * line_height, 'stats')
            self.render_text(screen, value_text, x + 150, y + 40 + i * line_height,
                           'stats', self.colors["highlight"])

    def get_text_size(self, text: str, font_type: str = 'ui') -> Tuple[int, int]:
        """
        Get the size of text when rendered with a specific font.

        Args:
            text: Text to measure
            font_type: Type of font to use

        Returns:
            Tuple of (width, height) in pixels
        """
        font = self.fonts.get(font_type, self.fonts['ui'])
        return font.size(text)

def render_game(screen: pygame.Surface, game_state: Dict) -> None:
    """
    Main function to render the complete game state.

    This function handles the rendering of all game elements including:
    - Player and dealer hands
    - Game statistics
    - Training information
    - UI controls

    Args:
        screen: Pygame surface to render to
        game_state: Dictionary containing the current game state with keys:
            - player_hand: List of player card dictionaries
            - dealer_hand: List of dealer card dictionaries
            - player_value: Player hand value
            - dealer_value: Dealer hand value
            - bankroll: Current bankroll
            - current_bet: Current bet amount
            - running_count: Hi-Lo running count
            - true_count: True count value
            - decks_remaining: Number of decks remaining
            - game_over: Whether hand is complete
            - player_bust: Whether player busted
            - dealer_bust: Whether dealer busted
            - player_blackjack: Whether player has blackjack
            - dealer_blackjack: Whether dealer has blackjack
            - hands_played: Total hands played
            - win_rate: Current win rate
            - exploration_rate: RL agent exploration rate
            - states_visited: Number of Q-table states visited
    """
    # Clear screen with background color
    screen.fill(config.VISUALIZATION_CONFIG["colors"]["background"])

    # Initialize renderers
    card_renderer = CardRenderer()
    text_renderer = TextRenderer()

    # Render title
    text_renderer.render_centered_text(
        screen, "Blackjack Reinforcement Learning Agent",
        config.VISUALIZATION_CONFIG["screen_width"] // 2,
        20, 'title'
    )

    # Render player and dealer hands
    _render_hands(screen, game_state, card_renderer, text_renderer)

    # Render game statistics
    _render_game_stats(screen, game_state, text_renderer)

    # Render training information
    _render_training_info(screen, game_state, text_renderer)

    # Render controls
    _render_controls(screen, text_renderer)

    # Update display
    pygame.display.flip()

def _render_hands(screen: pygame.Surface, game_state: Dict,
                 card_renderer: CardRenderer, text_renderer: TextRenderer) -> None:
    """
    Render player and dealer hands with cards and values.

    Args:
        screen: Pygame surface to render to
        game_state: Current game state
        card_renderer: CardRenderer instance
        text_renderer: TextRenderer instance
    """
    config_vis = config.VISUALIZATION_CONFIG
    card_width = config_vis["card_width"]
    card_height = config_vis["card_height"]
    card_spacing = config_vis["card_spacing"]

    # Player hand position
    player_hands = game_state.get('player_hand', [])
    if not isinstance(player_hands, list):
        player_hands = []

    player_x = config_vis["screen_width"] // 2 - (len(player_hands) * (card_width + card_spacing)) // 2
    player_y = config_vis["screen_height"] - card_height - 100

    # Render player hand
    text_renderer.render_text(screen, "Player:", player_x, player_y - 30, 'ui')

    for i, card in enumerate(player_hands):
        card_x = player_x + i * (card_width + card_spacing)

        # Ensure card is a dict with rank and suit
        if isinstance(card, dict) and 'rank' in card and 'suit' in card:
            card_renderer.render_card(screen, card, card_x, player_y, True)

            # Show card value
            if card.get('value'):
                text_renderer.render_centered_text(
                    screen, str(card['value']),
                    card_x + card_width // 2,
                    player_y + card_height + 5, 'small'
                )

    # Show player hand value
    player_value = game_state.get('player_value', 0)
    player_value_text = f"Value: {player_value}"
    if game_state.get('player_bust'):
        player_value_text += " (BUST)"
    elif game_state.get('player_blackjack'):
        player_value_text += " (BLACKJACK)"

    text_renderer.render_text(
        screen, player_value_text,
        player_x, player_y + card_height + 30, 'ui'
    )

    # Dealer hand position
    dealer_hands = game_state.get('dealer_hand', [])
    if not isinstance(dealer_hands, list):
        dealer_hands = []

    dealer_x = config_vis["screen_width"] // 2 - (len(dealer_hands) * (card_width + card_spacing)) // 2
    dealer_y = 150

    # Render dealer hand
    text_renderer.render_text(screen, "Dealer:", dealer_x, dealer_y - 30, 'ui')

    for i, card in enumerate(dealer_hands):
        card_x = dealer_x + i * (card_width + card_spacing)

        # First card is face down if game is not over
        face_up = True
        if i == 0 and not game_state.get('game_over', True):
            face_up = False

        if isinstance(card, dict) and 'rank' in card and 'suit' in card:
            card_renderer.render_card(screen, card, card_x, dealer_y, face_up)

            # Show card value if face up
            if face_up and card.get('value'):
                text_renderer.render_centered_text(
                    screen, str(card['value']),
                    card_x + card_width // 2,
                    dealer_y + card_height + 5, 'small'
                )

    # Show dealer hand value if game is over
    if game_state.get('game_over', False):
        dealer_value = game_state.get('dealer_value', 0)
        dealer_value_text = f"Value: {dealer_value}"
        if game_state.get('dealer_bust'):
            dealer_value_text += " (BUST)"
        elif game_state.get('dealer_blackjack'):
            dealer_value_text += " (BLACKJ