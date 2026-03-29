"""
solitaire.py - Core game logic for Klondike solitaire.

This module implements the fundamental classes and logic for a Klondike solitaire game,
including card representation, deck management, and game state handling.
"""

import random
from typing import List, Tuple, Optional

class Card:
    """
    Represents a playing card with suit, rank, position, and face-up status.

    Attributes:
        suit (str): The suit of the card (Hearts, Diamonds, Clubs, Spades).
        rank (str): The rank of the card (Ace, 2-10, Jack, Queen, King).
        position (tuple): The (x, y) position of the card on the screen.
        face_up (bool): Whether the card is face up or face down.
    """
    VALID_SUITS = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
    VALID_RANKS = ['Ace', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'Jack', 'Queen', 'King']

    def __init__(self, suit: str, rank: str, position: Tuple[int, int] = (0, 0), face_up: bool = False):
        """
        Initialize a new Card instance.

        Args:
            suit: The suit of the card.
            rank: The rank of the card.
            position: The initial position of the card.
            face_up: Whether the card is initially face up.
        """
        if suit not in self.VALID_SUITS:
            raise ValueError(f"Invalid suit: {suit}. Must be one of {self.VALID_SUITS}")
        if rank not in self.VALID_RANKS:
            raise ValueError(f"Invalid rank: {rank}. Must be one of {self.VALID_RANKS}")

        self.suit = suit
        self.rank = rank
        self.position = position
        self.face_up = face_up

    def __str__(self) -> str:
        """Return a string representation of the card."""
        return f"{self.rank} of {self.suit}"

    def __repr__(self) -> str:
        """Return a detailed string representation of the card."""
        return f"Card(suit='{self.suit}', rank='{self.rank}', position={self.position}, face_up={self.face_up})"

    def flip(self) -> None:
        """Flip the card's face-up status."""
        self.face_up = not self.face_up

    def get_value(self) -> int:
        """
        Get the numerical value of the card for game logic.

        Returns:
            The numerical value (Ace=1, 2-10=face value, Jack=11, Queen=12, King=13).
        """
        if self.rank == 'Ace':
            return 1
        elif self.rank in ['Jack', 'Queen', 'King']:
            return {'Jack': 11, 'Queen': 12, 'King': 13}[self.rank]
        else:
            return int(self.rank)

    def is_red(self) -> bool:
        """Check if the card is red (Hearts or Diamonds)."""
        return self.suit in ['Hearts', 'Diamonds']

    def is_black(self) -> bool:
        """Check if the card is black (Clubs or Spades)."""
        return self.suit in ['Clubs', 'Spades']

class Deck:
    """
    Represents a deck of playing cards with shuffling and drawing capabilities.

    Attributes:
        cards (List[Card]): The list of cards in the deck.
    """
    def __init__(self):
        """Initialize a new Deck with a standard 52-card deck."""
        self.cards = []
        for suit in Card.VALID_SUITS:
            for rank in Card.VALID_RANKS:
                self.cards.append(Card(suit, rank))

    def shuffle(self) -> None:
        """Shuffle the deck using Fisher-Yates algorithm."""
        random.shuffle(self.cards)

    def draw(self) -> Optional[Card]:
        """
        Draw the top card from the deck.

        Returns:
            The top card if the deck is not empty, None otherwise.
        """
        if len(self.cards) > 0:
            return self.cards.pop()
        return None

    def draw_multiple(self, num_cards: int) -> List[Card]:
        """
        Draw multiple cards from the top of the deck.

        Args:
            num_cards: The number of cards to draw.

        Returns:
            A list of the drawn cards. If the deck has fewer cards than requested,
            returns all remaining cards.
        """
        if num_cards <= 0:
            return []

        actual_num = min(num_cards, len(self.cards))
        drawn_cards = self.cards[-actual_num:]
        self.cards = self.cards[:-actual_num]
        return drawn_cards

    def add_card(self, card: Card, position: int = -1) -> None:
        """
        Add a card to the deck at a specific position.

        Args:
            card: The card to add.
            position: The position to insert the card (default: -1, end of deck).
        """
        if position == -1:
            self.cards.append(card)
        else:
            self.cards.insert(position, card)

    def add_cards(self, cards: List[Card], position: int = -1) -> None:
        """
        Add multiple cards to the deck at a specific position.

        Args:
            cards: The list of cards to add.
            position: The position to insert the cards (default: -1, end of deck).
        """
        if position == -1:
            self.cards.extend(cards)
        else:
            self.cards[position:position] = cards

    def is_empty(self) -> bool:
        """Check if the deck is empty."""
        return len(self.cards) == 0

    def __len__(self) -> int:
        """Return the number of cards in the deck."""
        return len(self.cards)

class SolitaireGame:
    """
    Implements the core logic for a Klondike solitaire game.

    Attributes:
        tableau (List[List[Card]]): The 7 tableau piles.
        foundations (List[List[Card]]): The 4 foundation piles.
        stock (Deck): The stock pile (face-down cards).
        waste (List[Card]): The waste pile (face-up cards from stock).
        score (int): The current game score.
    """
    def __init__(self):
        """Initialize a new solitaire game."""
        self.tableau = [[] for _ in range(7)]
        self.foundations = [[] for _ in range(4)]
        self.stock = Deck()
        self.waste = []
        self.score = 0
        self._initialize_game()

    def _initialize_game(self) -> None:
        """Set up the initial game state."""
        # Shuffle the deck
        self.stock.shuffle()

        # Deal cards to tableau
        for i in range(7):
            num_cards = i + 1
            cards = self.stock.draw_multiple(num_cards)
            for j, card in enumerate(cards):
                # Last card in each pile is face up, others face down
                card.face_up = (j == len(cards) - 1)
            self.tableau[i] = cards

    def deal_from_stock(self) -> bool:
        """
        Deal cards from the stock to the waste pile.

        Returns:
            True if cards were dealt, False if stock is empty.
        """
        if self.stock.is_empty():
            # If stock is empty, reset it from waste
            if len(self.waste) > 0:
                self.stock.add_cards(self.waste)
                self.stock.cards.reverse()  # Maintain original order
                self.waste = []
                return self.deal_from_stock()
            return False

        # Deal 3 cards (or remaining cards if less than 3)
        num_cards = min(3, len(self.stock))
        cards = self.stock.draw_multiple(num_cards)

        # Add to waste pile (face up)
        for card in cards:
            card.face_up = True
        self.waste.extend(cards)

        return True

    def can_move_to_foundation(self, card: Card, foundation_idx: int) -> bool:
        """
        Check if a card can be moved to a foundation pile.

        Args:
            card: The card to check.
            foundation_idx: The index of the foundation pile (0-3).

        Returns:
            True if the move is valid, False otherwise.
        """
        foundation = self.foundations[foundation_idx]

        # Foundation is empty - only Ace can be placed
        if len(foundation) == 0:
            return card.rank == 'Ace'

        top_card = foundation[-1]

        # Check suit matches foundation suit (0-3: Hearts, Diamonds, Clubs, Spades)
        expected_suit = Card.VALID_SUITS[foundation_idx]
        if card.suit != expected_suit:
            return False

        # Check rank is one higher than top card
        return card.get_value() == top_card.get_value() + 1

    def move_to_foundation(self, card: Card, foundation_idx: int) -> bool:
        """
        Move a card to a foundation pile.

        Args:
            card: The card to move.
            foundation_idx: The index of the foundation pile (0-3).

        Returns:
            True if the move was successful, False otherwise.
        """
        if not self.can_move_to_foundation(card, foundation_idx):
            return False

        # Remove card from its current location
        self._remove_card_from_piles(card)

        # Add to foundation
        self.foundations[foundation_idx].append(card)
        self.score += 10

        # Check if we won
        if self._check_win():
            self.score += 100

        return True

    def can_move_to_tableau(self, cards: List[Card], tableau_idx: int) -> bool:
        """
        Check if a sequence of cards can be moved to a tableau pile.

        Args:
            cards: The list of cards to move (must be in order).
            tableau_idx: The index of the target tableau pile (0-6).

        Returns:
            True if the move is valid, False otherwise.
        """
        target_pile = self.tableau[tableau_idx]

        # If target pile is empty, only King can be placed
        if len(target_pile) == 0:
            return len(cards) == 1 and cards[0].rank == 'King'

        target_card = target_pile[-1]

        # Check bottom card of sequence (cards[0]) can be placed on target
        bottom_card = cards[0]

        # Colors must be different
        if bottom_card.is_red() == target_card.is_red():
            return False

        # Ranks must be in descending order (bottom_card is one less than target)
        return bottom_card.get_value() == target_card.get_value() - 1

    def move_to_tableau(self, cards: List[Card], tableau_idx: int) -> bool:
        """
        Move a sequence of cards to a tableau pile.

        Args:
            cards: The list of cards to move.
            tableau_idx: The index of the target tableau pile (0-6).

        Returns:
            True if the move was successful, False otherwise.
        """
        if not self.can_move_to_tableau(cards, tableau_idx):
            return False

        # Remove cards from their current location
        for card in cards:
            self._remove_card_from_piles(card)

        # Add to target tableau pile
        self.tableau[tableau_idx].extend(cards)
        self.score += 5

        return True

    def _remove_card_from_piles(self, card: Card) -> bool:
        """
        Remove a card from any pile it might be in.

        Args:
            card: The card to remove.

        Returns:
            True if the card was found and removed, False otherwise.
        """
        # Check waste pile
        if card in self.waste:
            self.waste.remove(card)
            return True

        # Check tableau piles
        for pile in self.tableau:
            if card in pile:
                index = pile.index(card)
                pile.remove(card)

                # If we removed a face-up card and there are cards left in the pile,
                # flip the new top card if it was face down
                if index < len(pile) and not pile[index].face_up:
                    pile[index].flip()
                return True

        # Card not found in any pile
        return False

    def _check_win(self) -> bool:
        """
        Check if the game has been won (all cards in foundations).

        Returns:
            True if the game is won, False otherwise.
        """
        # Each foundation should have 13 cards (Ace through King)
        return all(len(foundation) == 13 for foundation in self.foundations)

    def get_valid_moves(self) -> List[Tuple]:
        """
        Get all valid moves in the current game state.

        Returns:
            A list of tuples representing valid moves. Each tuple contains:
            - Move type ('to_foundation' or 'to_tableau')
            - Source information (card or list of cards)
            - Destination information (foundation index or tableau index)
        """
        valid_moves = []

        # Check moves from waste to foundations
        if len(self.waste) > 0:
            top_waste = self.waste[-1]
            for i in range(4):
                if self.can_move_to_foundation(top_waste, i):
                    valid_moves.append(('to_foundation', top_waste, i))

        # Check moves from waste to tableau
        if len(self.waste) > 0:
            top_waste = self.waste[-1]
            for i in range(7):
                if self.can_move_to_tableau([top_waste], i):
                    valid_moves.append(('to_tableau', [top_waste], i))

        # Check moves from tableau to foundations
        for pile_idx, pile in enumerate(self.tableau):
            if len(pile) > 0 and pile[-1].face_up:
                top_card = pile[-1]
                for foundation_idx in range(4):
                    if self.can_move_to_foundation(top_card, foundation_idx):
                        valid_moves.append(('to_foundation', top_card, foundation_idx))

        # Check moves from tableau to tableau
        for source_idx, source_pile in enumerate(self.tableau):
            if len(source_pile) == 0:
                continue

            # Find the first face-up card in the pile
            first_face_up = None
            for i, card in enumerate(source_pile):
                if card.face_up:
                    first_face_up = i
                    break

            if first_face_up is None:
                continue

            # Try moving sequences of different lengths
            for seq_length in range(1, len(source_pile) - first_face_up + 1):
                sequence = source_pile[first_face_up:first_face_up + seq_length]

                for target_idx in range(7):
                    if source_idx != target_idx and self.can_move_to_tableau(sequence, target_idx):
                        valid_moves.append(('to_tableau', sequence, target_idx))

        return valid_moves

    def apply_move(self, move: Tuple) -> int:
        """
        Apply a move to the game state.

        Args:
            move: A tuple representing the move to apply. Format:
                - ('to_foundation', card, foundation_idx)
                - ('to_tableau', [cards], tableau_idx)

        Returns:
            The reward for the move (positive for valid, negative for invalid).
        """
        move_type = move[0]

        if move_type == 'to_foundation':
            card = move[1]
            foundation_idx = move[2]
            if self.move_to_foundation(card, foundation_idx):
                return 10
            return -1

        elif move_type == 'to_tableau':
            cards = move[1]
            tableau_idx = move[2]
            if self.move_to_tableau(cards, tableau_idx):
                return 5
            return -1

        return -1

    def reset(self) -> None:
        """Reset the game to its initial state."""
        self.__init__()