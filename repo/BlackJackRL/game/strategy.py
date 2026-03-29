"""
Basic Strategy Implementation for Blackjack.

This module implements the BasicStrategy class which provides optimal playing decisions
based on the player's hand and the dealer's upcard. The strategy follows standard blackjack
basic strategy rules to maximize expected value for each possible hand combination.

The strategy table is based on standard blackjack rules where:
- Dealer stands on soft 17
- Double after split is allowed
- No resplitting of aces
- No surrender

The strategy covers all possible player hands (hard totals, soft totals, and pairs)
against all possible dealer upcards (2 through Ace).
"""

from typing import List, Dict, Union
from game.blackjack import Card

class BasicStrategy:
    """
    Implements basic strategy for blackjack playing decisions.

    This class provides methods to determine the optimal action (hit, stand, double, split)
    based on the player's hand and the dealer's upcard. The strategy follows standard
    blackjack basic strategy to maximize expected value.

    Attributes:
        strategy_table (dict): A nested dictionary containing the optimal strategy for all
            possible player hand and dealer upcard combinations.
    """

    def __init__(self):
        """
        Initializes the BasicStrategy class with a predefined strategy table.

        The strategy table is structured as:
        {
            'hard': {player_total: {dealer_upcard: action}},
            'soft': {player_total: {dealer_upcard: action}},
            'pairs': {card_rank: {dealer_upcard: action}}
        }
        """
        self.strategy_table = {
            'hard': self._create_hard_totals_table(),
            'soft': self._create_soft_totals_table(),
            'pairs': self._create_pairs_table()
        }

    def _create_hard_totals_table(self) -> dict:
        """
        Creates the strategy table for hard totals (no aces or aces counted as 1).

        Returns:
            dict: A nested dictionary mapping player totals and dealer upcards to actions.
        """
        return {
            5: {2: 'H', 3: 'H', 4: 'H', 5: 'H', 6: 'H', 7: 'H', 8: 'H', 9: 'H', 10: 'H', 'A': 'H'},
            6: {2: 'H', 3: 'H', 4: 'H', 5: 'H', 6: 'H', 7: 'H', 8: 'H', 9: 'H', 10: 'H', 'A': 'H'},
            7: {2: 'H', 3: 'H', 4: 'H', 5: 'H', 6: 'H', 7: 'H', 8: 'H', 9: 'H', 10: 'H', 'A': 'H'},
            8: {2: 'H', 3: 'H', 4: 'H', 5: 'H', 6: 'H', 7: 'H', 8: 'H', 9: 'H', 10: 'H', 'A': 'H'},
            9: {2: 'H', 3: 'D', 4: 'D', 5: 'D', 6: 'D', 7: 'H', 8: 'H', 9: 'H', 10: 'H', 'A': 'H'},
            10: {2: 'D', 3: 'D', 4: 'D', 5: 'D', 6: 'D', 7: 'D', 8: 'D', 9: 'D', 10: 'H', 'A': 'H'},
            11: {2: 'D', 3: 'D', 4: 'D', 5: 'D', 6: 'D', 7: 'D', 8: 'D', 9: 'D', 10: 'D', 'A': 'H'},
            12: {2: 'H', 3: 'H', 4: 'S', 5: 'S', 6: 'S', 7: 'H', 8: 'H', 9: 'H', 10: 'H', 'A': 'H'},
            13: {2: 'S', 3: 'S', 4: 'S', 5: 'S', 6: 'S', 7: 'H', 8: 'H', 9: 'H', 10: 'H', 'A': 'H'},
            14: {2: 'S', 3: 'S', 4: 'S', 5: 'S', 6: 'S', 7: 'H', 8: 'H', 9: 'H', 10: 'H', 'A': 'H'},
            15: {2: 'S', 3: 'S', 4: 'S', 5: 'S', 6: 'S', 7: 'H', 8: 'H', 9: 'H', 10: 'H', 'A': 'H'},
            16: {2: 'S', 3: 'S', 4: 'S', 5: 'S', 6: 'S', 7: 'H', 8: 'H', 9: 'H', 10: 'H', 'A': 'H'},
            17: {2: 'S', 3: 'S', 4: 'S', 5: 'S', 6: 'S', 7: 'S', 8: 'S', 9: 'S', 10: 'S', 'A': 'S'},
            18: {2: 'S', 3: 'S', 4: 'S', 5: 'S', 6: 'S', 7: 'S', 8: 'S', 9: 'S', 10: 'S', 'A': 'S'},
            19: {2: 'S', 3: 'S', 4: 'S', 5: 'S', 6: 'S', 7: 'S', 8: 'S', 9: 'S', 10: 'S', 'A': 'S'},
            20: {2: 'S', 3: 'S', 4: 'S', 5: 'S', 6: 'S', 7: 'S', 8: 'S', 9: 'S', 10: 'S', 'A': 'S'},
            21: {2: 'S', 3: 'S', 4: 'S', 5: 'S', 6: 'S', 7: 'S', 8: 'S', 9: 'S', 10: 'S', 'A': 'S'}
        }

    def _create_soft_totals_table(self) -> dict:
        """
        Creates the strategy table for soft totals (hands containing an ace counted as 11).

        Returns:
            dict: A nested dictionary mapping player totals and dealer upcards to actions.
        """
        return {
            13: {2: 'H', 3: 'H', 4: 'D', 5: 'D', 6: 'D', 7: 'H', 8: 'H', 9: 'H', 10: 'H', 'A': 'H'},
            14: {2: 'H', 3: 'H', 4: 'D', 5: 'D', 6: 'D', 7: 'H', 8: 'H', 9: 'H', 10: 'H', 'A': 'H'},
            15: {2: 'H', 3: 'H', 4: 'D', 5: 'D', 6: 'D', 7: 'H', 8: 'H', 9: 'H', 10: 'H', 'A': 'H'},
            16: {2: 'H', 3: 'H', 4: 'D', 5: 'D', 6: 'D', 7: 'H', 8: 'H', 9: 'H', 10: 'H', 'A': 'H'},
            17: {2: 'H', 3: 'D', 4: 'D', 5: 'D', 6: 'D', 7: 'H', 8: 'H', 9: 'H', 10: 'H', 'A': 'H'},
            18: {2: 'S', 3: 'D', 4: 'D', 5: 'D', 6: 'D', 7: 'S', 8: 'S', 9: 'H', 10: 'H', 'A': 'S'},
            19: {2: 'S', 3: 'S', 4: 'S', 5: 'S', 6: 'S', 7: 'S', 8: 'S', 9: 'S', 10: 'S', 'A': 'S'},
            20: {2: 'S', 3: 'S', 4: 'S', 5: 'S', 6: 'S', 7: 'S', 8: 'S', 9: 'S', 10: 'S', 'A': 'S'},
            21: {2: 'S', 3: 'S', 4: 'S', 5: 'S', 6: 'S', 7: 'S', 8: 'S', 9: 'S', 10: 'S', 'A': 'S'}
        }

    def _create_pairs_table(self) -> dict:
        """
        Creates the strategy table for pairs (hands with two cards of the same rank).

        Returns:
            dict: A nested dictionary mapping card ranks and dealer upcards to actions.
        """
        return {
            2: {2: 'P', 3: 'P', 4: 'P', 5: 'P', 6: 'P', 7: 'P', 8: 'H', 9: 'H', 10: 'H', 'A': 'H'},
            3: {2: 'P', 3: 'P', 4: 'P', 5: 'P', 6: 'P', 7: 'P', 8: 'H', 9: 'H', 10: 'H', 'A': 'H'},
            4: {2: 'H', 3: 'H', 4: 'H', 5: 'P', 6: 'P', 7: 'H', 8: 'H', 9: 'H', 10: 'H', 'A': 'H'},
            5: {2: 'D', 3: 'D', 4: 'D', 5: 'D', 6: 'D', 7: 'D', 8: 'D', 9: 'D', 10: 'H', 'A': 'H'},
            6: {2: 'P', 3: 'P', 4: 'P', 5: 'P', 6: 'P', 7: 'H', 8: 'H', 9: 'H', 10: 'H', 'A': 'H'},
            7: {2: 'P', 3: 'P', 4: 'P', 5: 'P', 6: 'P', 7: 'P', 8: 'H', 9: 'H', 10: 'H', 'A': 'H'},
            8: {2: 'P', 3: 'P', 4: 'P', 5: 'P', 6: 'P', 7: 'P', 8: 'P', 9: 'P', 10: 'P', 'A': 'P'},
            9: {2: 'P', 3: 'P', 4: 'P', 5: 'P', 6: 'P', 7: 'S', 8: 'P', 9: 'P', 10: 'S', 'A': 'S'},
            10: {2: 'S', 3: 'S', 4: 'S', 5: 'S', 6: 'S', 7: 'S', 8: 'S', 9: 'S', 10: 'S', 'A': 'S'},
            'A': {2: 'P', 3: 'P', 4: 'P', 5: 'P', 6: 'P', 7: 'P', 8: 'P', 9: 'P', 10: 'P', 'A': 'P'}
        }

    def get_action(self, player_hand: List[Card], dealer_upcard: Card) -> str:
        """
        Determines the optimal action based on the player's hand and dealer's upcard.

        Args:
            player_hand: List of Card objects representing the player's hand.
            dealer_upcard: The dealer's face-up card.

        Returns:
            str: The optimal action ('H' for Hit, 'S' for Stand, 'D' for Double, 'P' for Split).

        Raises:
            ValueError: If the hand or dealer upcard is invalid.
        """
        if not player_hand or not dealer_upcard:
            raise ValueError("Invalid hand or dealer upcard")

        # Check if hand is a pair
        if len(player_hand) == 2 and player_hand[0].rank == player_hand[1].rank:
            return self._get_pair_action(player_hand[0].rank, dealer_upcard)

        # Check if hand is soft (contains an ace counted as 11)
        if self._is_soft_hand(player_hand):
            return self._get_soft_action(player_hand, dealer_upcard)

        # Otherwise, treat as hard total
        return self._get_hard_action(player_hand, dealer_upcard)

    def _get_pair_action(self, card_rank: str, dealer_upcard: Card) -> str:
        """
        Gets the action for a pair hand.

        Args:
            card_rank: The rank of the paired cards.
            dealer_upcard: The dealer's face-up card.

        Returns:
            str: The optimal action.
        """
        # Convert card rank to numeric value for lookup
        rank_value = card_rank
        if card_rank in ['J', 'Q', 'K']:
            rank_value = 10
        elif card_rank == 'A':
            rank_value = 'A'

        # Get dealer upcard value
        dealer_value = self._get_card_value(dealer_upcard)

        return self.strategy_table['pairs'][rank_value][dealer_value]

    def _get_soft_action(self, player_hand: List[Card], dealer_upcard: Card) -> str:
        """
        Gets the action for a soft hand (contains an ace counted as 11).

        Args:
            player_hand: List of Card objects.
            dealer_upcard: The dealer's face-up card.

        Returns:
            str: The optimal action.
        """
        # Calculate soft total
        total = self._calculate_hand_total(player_hand)

        # Get dealer upcard value
        dealer_value = self._get_card_value(dealer_upcard)

        # Handle special case for A,2 (soft 13)
        if total == 13 and len(player_hand) == 2 and any(card.rank == 'A' for card in player_hand):
            return self.strategy_table['soft'][13][dealer_value]

        # For other soft totals
        if total in self.strategy_table['soft']:
            return self.strategy_table['soft'][total][dealer_value]
        else:
            # If soft total is not in table (e.g., A,9 = soft 20), stand
            return 'S'

    def _get_hard_action(self, player_hand: List[Card], dealer_upcard: Card) -> str:
        """
        Gets the action for a hard hand (no aces or aces counted as 1).

        Args:
            player_hand: List of Card objects.
            dealer_upcard: The dealer's face-up card.

        Returns:
            str: The optimal action.
        """
        # Calculate hard total
        total = self._calculate_hand_total(player_hand)

        # Get dealer upcard value
        dealer_value = self._get_card_value(dealer_upcard)

        # Handle special case for 21 (always stand)
        if total == 21:
            return 'S'

        # For other hard totals
        if total in self.strategy_table['hard']:
            return self.strategy_table['hard'][total][dealer_value]
        else:
            # If total is not in table (shouldn't happen for valid hands), stand
            return 'S'

    def _is_soft_hand(self, player_hand: List[Card]) -> bool:
        """
        Checks if a hand is a soft hand (contains an ace counted as 11).

        Args:
            player_hand: List of Card objects.

        Returns:
            bool: True if the hand is soft, False otherwise.
        """
        # Calculate total with aces as 11
        total = sum(card.value for card in player_hand)

        # Check if there's an ace and total is <= 21
        has_ace = any(card.rank == 'A' for card in player_hand)
        return has_ace and total <= 21

    def _calculate_hand_total(self, player_hand: List[Card]) -> int:
        """
        Calculates the total value of a hand, accounting for aces.

        Args:
            player_hand: List of Card objects.

        Returns:
            int: The total