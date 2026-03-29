"""
Core blackjack game engine for the Blackjack Reinforcement Learning Agent.

This module contains the Card, Deck, and BlackjackGame classes that implement
the fundamental game logic, card handling, and deck management for a blackjack
game with Hi-Lo card counting capabilities.
"""

import random
import math
from typing import List, Dict, Tuple, Optional
import config

class Card:
    """
    Represents a single playing card in blackjack.

    Attributes:
        rank (str): The card's rank (2-10, J, Q, K, A)
        suit (str): The card's suit (Hearts, Diamonds, Clubs, Spades)
        value (int): The card's value in blackjack (2-11)
        count_value (int): The card's value in the Hi-Lo counting system
    """

    def __init__(self, rank: str, suit: str):
        """
        Initialize a Card with rank and suit.

        Args:
            rank: The card's rank (2-10, J, Q, K, A)
            suit: The card's suit (Hearts, Diamonds, Clubs, Spades)
        """
        self.rank = rank
        self.suit = suit
        self.value = self._get_value(rank)
        self.count_value = config.COUNTING_CONFIG["card_values"].get(rank, 0)

    def _get_value(self, rank: str) -> int:
        """
        Get the blackjack value of a card based on its rank.

        Args:
            rank: The card's rank

        Returns:
            The card's value (2-11)
        """
        if rank in ['J', 'Q', 'K']:
            return 10
        elif rank == 'A':
            return 11  # Default to 11, will adjust if needed
        else:
            return int(rank)

    def __str__(self) -> str:
        """String representation of the card."""
        return f"{self.rank} of {self.suit}"

    def __repr__(self) -> str:
        """Detailed string representation of the card."""
        return f"Card(rank='{self.rank}', suit='{self.suit}', value={self.value})"

class Deck:
    """
    Represents a deck of playing cards for blackjack.

    Attributes:
        cards (List[Card]): List of Card objects in the deck
        count (int): Running count using Hi-Lo system
        num_decks (int): Number of decks in this shoe
    """

    def __init__(self, num_decks: int = 1):
        """
        Initialize a deck with the specified number of decks.

        Args:
            num_decks: Number of standard 52-card decks to include
        """
        self.num_decks = num_decks
        self.cards = []
        self.count = 0

        # Create the deck
        self._create_deck()
        self.shuffle()

    def _create_deck(self) -> None:
        """Create a new deck with the specified number of decks."""
        self.cards = []
        self.count = 0

        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']

        for _ in range(self.num_decks):
            for suit in suits:
                for rank in ranks:
                    self.cards.append(Card(rank, suit))

    def shuffle(self) -> None:
        """Shuffle the deck and reset the count."""
        random.shuffle(self.cards)
        self.count = 0

    def deal_card(self) -> Optional[Card]:
        """
        Deal a card from the top of the deck.

        Returns:
            The dealt Card object, or None if the deck is empty
        """
        if len(self.cards) == 0:
            return None

        card = self.cards.pop(0)
        self.count += card.count_value
        return card

    def cards_remaining(self) -> int:
        """
        Get the number of cards remaining in the deck.

        Returns:
            Number of cards remaining
        """
        return len(self.cards)

    def decks_remaining(self) -> float:
        """
        Estimate the number of decks remaining.

        Returns:
            Approximate number of decks remaining
        """
        return len(self.cards) / 52.0

    def penetration_reached(self, penetration: float) -> bool:
        """
        Check if the penetration threshold has been reached.

        Args:
            penetration: The penetration threshold (0-1)

        Returns:
            True if penetration threshold has been reached, False otherwise
        """
        if penetration <= 0 or penetration > 1:
            return False

        total_cards = self.num_decks * 52
        cards_dealt = total_cards - len(self.cards)
        return cards_dealt / total_cards >= penetration

def calculate_true_count(running_count: int, decks_remaining: float) -> float:
    """
    Calculate the true count based on running count and decks remaining.

    Args:
        running_count: The current running count
        decks_remaining: Number of decks remaining

    Returns:
        The true count (running count divided by decks remaining)
    """
    if decks_remaining <= 0:
        return 0.0
    return running_count / decks_remaining

class BlackjackGame:
    """
    Main blackjack game engine that handles game logic and state.

    Attributes:
        decks (List[Deck]): List of Deck objects (typically one shoe)
        player_hand (List[Card]): Current player hand
        dealer_hand (List[Card]): Current dealer hand
        deck_count (int): Current running count
        true_count (float): Current true count
        bankroll (float): Current player bankroll
        min_bet (float): Minimum bet size
        max_bet (float): Maximum bet size
        current_bet (float): Current bet for this hand
        penetration (float): Penetration threshold for reshuffling
        dealer_stands_on_soft_17 (bool): Whether dealer stands on soft 17
        double_after_split (bool): Whether doubling after split is allowed
        resplit_aces (bool): Whether resplitting aces is allowed
        late_surrender (bool): Whether late surrender is allowed
        game_over (bool): Whether the current hand is complete
    """

    def __init__(self, num_decks: int = 6, initial_bankroll: float = 1000.0,
                 min_bet: float = 10.0, max_bet: float = 120.0,
                 penetration: float = 0.75):
        """
        Initialize a new blackjack game.

        Args:
            num_decks: Number of decks to use
            initial_bankroll: Starting bankroll
            min_bet: Minimum bet size
            max_bet: Maximum bet size
            penetration: Penetration threshold for reshuffling
        """
        # Game configuration
        self.num_decks = num_decks
        self.initial_bankroll = initial_bankroll
        self.min_bet = min_bet
        self.max_bet = max_bet
        self.penetration = penetration

        # Load additional config from config module
        game_config = config.GAME_CONFIG
        self.dealer_stands_on_soft_17 = game_config["dealer_stands_on_soft_17"]
        self.double_after_split = game_config["double_after_split"]
        self.resplit_aces = game_config["resplit_aces"]
        self.late_surrender = game_config["late_surrender"]

        # Initialize game state
        self.decks = [Deck(num_decks)]
        self.player_hand = []
        self.dealer_hand = []
        self.deck_count = 0
        self.true_count = 0.0
        self.bankroll = initial_bankroll
        self.current_bet = 0.0
        self.game_over = False

    def reset_game(self) -> None:
        """Reset the game state for a new hand."""
        self.player_hand = []
        self.dealer_hand = []
        self.current_bet = 0.0
        self.game_over = False

    def check_shuffle(self) -> bool:
        """
        Check if the deck needs to be shuffled based on penetration.

        Returns:
            True if shuffle is needed, False otherwise
        """
        if self.decks[0].penetration_reached(self.penetration):
            self.decks[0].shuffle()
            self.deck_count = 0
            return True
        return False

    def place_bet(self, bet_size: float) -> bool:
        """
        Place a bet for the current hand.

        Args:
            bet_size: The amount to bet

        Returns:
            True if bet was successful, False otherwise
        """
        if bet_size < self.min_bet or bet_size > self.max_bet:
            return False

        if bet_size > self.bankroll:
            return False

        self.current_bet = bet_size
        self.bankroll -= bet_size
        return True

    def deal_initial_cards(self) -> None:
        """Deal the initial two cards to player and dealer."""
        # Clear any existing hands
        self.player_hand = []
        self.dealer_hand = []

        # Deal cards
        for _ in range(2):
            self.player_hand.append(self._deal_card())
            self.dealer_hand.append(self._deal_card())

    def _deal_card(self) -> Card:
        """
        Deal a card from the deck and update counts.

        Returns:
            The dealt Card object
        """
        card = self.decks[0].deal_card()
        if card:
            self.deck_count += card.count_value
            decks_remaining = self.decks[0].decks_remaining()
            self.true_count = calculate_true_count(self.deck_count, decks_remaining)
        return card

    def get_hand_value(self, hand: List[Card]) -> int:
        """
        Calculate the value of a blackjack hand.

        Args:
            hand: List of Card objects

        Returns:
            The total value of the hand (aces counted as 11 or 1 as appropriate)
        """
        value = 0
        aces = 0

        for card in hand:
            value += card.value
            if card.rank == 'A':
                aces += 1

        # Adjust for aces
        while value > 21 and aces > 0:
            value -= 10
            aces -= 1

        return value

    def is_blackjack(self, hand: List[Card]) -> bool:
        """
        Check if a hand is a blackjack (ace + 10-value card).

        Args:
            hand: List of Card objects

        Returns:
            True if the hand is a blackjack, False otherwise
        """
        if len(hand) != 2:
            return False

        values = [card.value for card in hand]
        return 11 in values and 10 in values

    def is_bust(self, hand: List[Card]) -> bool:
        """
        Check if a hand is a bust (value > 21).

        Args:
            hand: List of Card objects

        Returns:
            True if the hand is a bust, False otherwise
        """
        return self.get_hand_value(hand) > 21

    def is_soft(self, hand: List[Card]) -> bool:
        """
        Check if a hand is soft (contains an ace counted as 11).

        Args:
            hand: List of Card objects

        Returns:
            True if the hand is soft, False otherwise
        """
        value = self.get_hand_value(hand)
        # Check if there's an ace that could be counted as 1
        has_ace = any(card.rank == 'A' for card in hand)
        return has_ace and value <= 11

    def play_hand(self, strategy: 'BasicStrategy') -> float:
        """
        Play a complete blackjack hand using the specified strategy.

        Args:
            strategy: BasicStrategy object to use for playing decisions

        Returns:
            The net gain/loss from the hand
        """
        # Reset game state
        self.reset_game()

        # Check if we need to shuffle
        self.check_shuffle()

        # Get bet size (this would normally come from the RL agent)
        # For now, we'll use a simple bet based on true count
        bet_size = self._get_simple_bet()
        self.place_bet(bet_size)

        # Deal initial cards
        self.deal_initial_cards()

        # Check for blackjack
        player_bj = self.is_blackjack(self.player_hand)
        dealer_bj = self.is_blackjack(self.dealer_hand)

        if player_bj:
            if dealer_bj:
                # Push
                self.bankroll += self.current_bet
                self.game_over = True
                return 0.0
            else:
                # Player wins with blackjack
                payout = self.current_bet * config.GAME_CONFIG["blackjack_payout"]
                self.bankroll += self.current_bet + payout
                self.game_over = True
                return payout
        elif dealer_bj:
            # Dealer wins with blackjack
            self.game_over = True
            return -self.current_bet

        # Player's turn
        while not self.game_over:
            # Get player action from strategy
            player_value = self.get_hand_value(self.player_hand)
            dealer_upcard = self.dealer_hand[0].value

            action = strategy.get_action(player_value, dealer_upcard,
                                        self.is_soft(self.player_hand),
                                        len(self.player_hand) == 2)

            if action == 'stand':
                break
            elif action == 'hit':
                self.player_hand.append(self._deal_card())
                if self.is_bust(self.player_hand):
                    self.game_over = True
                    return -self.current_bet
            elif action == 'double':
                if self.bankroll >= self.current_bet:
                    self.bankroll -= self.current_bet
                    self.current_bet *= 2
                    self.player_hand.append(self._deal_card())
                    if self.is_bust(self.player_hand):
                        self.game_over = True
                        return -self.current_bet
                    break
                else:
                    # Can't double, just hit
                    self.player_hand.append(self._deal_card())
                    if self.is_bust(self.player_hand):
                        self.game_over = True
                        return -self.current_bet
            elif action == 'split':
                # For simplicity, we'll just play one hand
                # In a full implementation, we'd need to handle multiple hands
                pass

        # Dealer's turn
        if not self.game_over:
            dealer_value = self.get_hand_value(self.dealer_hand)
            while dealer_value < 17 or (dealer_value == 17 and not self.is_soft(self.dealer_hand) and not self.dealer_stands_on_soft_17):
                self.dealer_hand.append(self._deal_card())
                dealer_value = self.get_hand_value(self.dealer_hand)
                if self.is_bust(self.dealer_hand):
                    break

        # Determine outcome
        player_value = self.get_hand_value(self.player_hand)
        dealer_value = self.get_hand_value(self.dealer_hand)

        if self.is_bust(self.player_hand):
            self.game_over = True
            return -self.current_bet
        elif self.is_bust(self.dealer_hand):
            self.bankroll += self.current_bet * 2
            self.game_over = True
            return self.current_bet
        elif player_value > dealer_value:
            self.bankroll += self.current_bet * 2
            self.game_over = True
            return self.current_bet
        elif player_value < dealer_value:
            self.game_over = True
            return -self.current_bet
        else:
            # Push
            self.bankroll += self.current_bet
            self.game_over = True
            return 0.0

    def _get_simple_bet(self) -> float:
        """
        Simple bet sizing based on true count (for testing).
        In the full implementation, this would be handled by the RL agent.

        Returns:
            Bet size based on true count
        """
        true_count = self.true_count
        min_bet = self.min_bet
        max_bet = self.max_bet

        # Simple linear scaling based on true count
        if true_count <= 1:
            return min_bet
        elif true_count >= 7:
            return max_bet
        else:
            # Scale between min and max bet
            bet_multiplier = 1 + (true_count - 1) * (12 - 1) / (7 - 1)
            return min(max_bet, max(min_bet, min_bet * bet_multiplier))

    def get_game_state(self) -> Dict:
        """
        Get the current game state for visualization or RL purposes.

        Returns:
            Dictionary containing the current game state
        """
        return {
            'player_hand': self.player_hand,
            'dealer_hand': self.dealer_hand,
            'player_value': self.get_hand_value(self.player_hand),
            'dealer_value': self.get_hand_value(self.dealer_hand),
            'bankroll': self.bankroll,
            'current_bet': self.current_bet,
            'running_count': self.deck_count,
            'true_count': self.true_count,
            'decks_remaining': self.decks[0].decks_remaining(),
            'game_over': self.game_over,
            'player_bust': self.is_bust(self.player_hand),
            'dealer_bust': self.is_bust(self.dealer_hand),
            'player_blackjack': self.is_blackjack(self.player_hand),
            'dealer_blackjack': self.is_blackjack(self.dealer_hand)
        }