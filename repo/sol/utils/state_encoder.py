"""
state_encoder.py - Game state to vector conversion for the RL agent.

This module implements functions to convert the solitaire game state into a
vectorized representation suitable for the neural network. The encoding includes
card positions, suits, ranks, and face-up status, with padding for variable-length piles.
"""

import numpy as np
from game.solitaire import SolitaireGame, Card
import config

# Constants for state encoding
MAX_TABLEAU_CARDS = 19  # Maximum cards in any tableau pile
MAX_FOUNDATION_CARDS = 13  # Maximum cards in any foundation pile
MAX_WASTE_CARDS = 3  # Maximum visible cards in waste pile

# Feature dimensions
SUIT_DIM = 4  # One-hot for 4 suits
RANK_DIM = 13  # One-hot for 13 ranks
FACE_UP_DIM = 1  # Binary face-up status
POSITION_DIM = 1  # Normalized position in pile
PILE_TYPE_DIM = 3  # One-hot for pile type (tableau, foundation, waste)
PILE_INDEX_DIM = 7  # One-hot for tableau pile index (0-6)

# Total features per card
CARD_FEATURES = SUIT_DIM + RANK_DIM + FACE_UP_DIM + POSITION_DIM + PILE_TYPE_DIM + PILE_INDEX_DIM

def get_state_representation(game: SolitaireGame) -> np.ndarray:
    """
    Convert the game state into a vectorized representation for the neural network.

    The state representation includes:
    - Tableau piles (7 piles, each with up to MAX_TABLEAU_CARDS cards)
    - Foundation piles (4 piles, each with up to MAX_FOUNDATION_CARDS cards)
    - Stock pile (remaining cards count)
    - Waste pile (up to MAX_WASTE_CARDS visible cards)

    Each card is represented by:
    - Suit (one-hot encoded)
    - Rank (one-hot encoded)
    - Face-up status (binary)
    - Position in pile (normalized)
    - Pile type (one-hot encoded)
    - Pile index (one-hot encoded for tableau piles)

    Args:
        game: The SolitaireGame instance to encode.

    Returns:
        A numpy array representing the game state.
    """
    # Calculate total state vector size
    tableau_size = 7 * MAX_TABLEAU_CARDS * CARD_FEATURES
    foundation_size = 4 * MAX_FOUNDATION_CARDS * CARD_FEATURES
    stock_size = 1
    waste_size = MAX_WASTE_CARDS * CARD_FEATURES

    total_size = tableau_size + foundation_size + stock_size + waste_size

    # Initialize state vector with zeros
    state_vector = np.zeros(total_size, dtype=np.float32)
    current_index = 0

    # Helper function to encode a card
    def encode_card(card, position_in_pile, max_pile_size, pile_type, pile_idx):
        """Encode a single card into the state vector."""
        if card is None:
            return current_index

        # Suit one-hot encoding
        suit_index = ['Hearts', 'Diamonds', 'Clubs', 'Spades'].index(card.suit)
        state_vector[current_index + suit_index] = 1.0

        # Rank one-hot encoding
        rank_value = Card.VALID_RANKS.index(card.rank)
        state_vector[current_index + SUIT_DIM + rank_value] = 1.0

        # Face-up status
        state_vector[current_index + SUIT_DIM + RANK_DIM] = 1.0 if card.face_up else 0.0

        # Position in pile (normalized)
        state_vector[current_index + SUIT_DIM + RANK_DIM + FACE_UP_DIM] = position_in_pile / max_pile_size

        # Pile type one-hot encoding
        pile_type_index = ['tableau', 'foundation', 'waste'].index(pile_type)
        state_vector[current_index + SUIT_DIM + RANK_DIM + FACE_UP_DIM + POSITION_DIM + pile_type_index] = 1.0

        # Pile index (for tableau piles)
        if pile_type == 'tableau':
            state_vector[current_index + SUIT_DIM + RANK_DIM + FACE_UP_DIM + POSITION_DIM + PILE_TYPE_DIM + pile_idx] = 1.0

        return current_index + CARD_FEATURES

    # Encode tableau piles
    for pile_idx in range(7):
        pile = game.tableau[pile_idx]
        num_cards = len(pile)

        for card_idx in range(MAX_TABLEAU_CARDS):
            if card_idx < num_cards:
                card = pile[card_idx]
                current_index = encode_card(card, card_idx, MAX_TABLEAU_CARDS, 'tableau', pile_idx)
            else:
                current_index += CARD_FEATURES

    # Encode foundation piles
    for pile_idx in range(4):
        pile = game.foundations[pile_idx]
        num_cards = len(pile)

        for card_idx in range(MAX_FOUNDATION_CARDS):
            if card_idx < num_cards:
                card = pile[card_idx]
                current_index = encode_card(card, card_idx, MAX_FOUNDATION_CARDS, 'foundation', 0)
            else:
                current_index += CARD_FEATURES

    # Encode stock pile (just the count, normalized)
    stock_count = len(game.stock.cards)
    state_vector[current_index] = stock_count / 52.0
    current_index += stock_size

    # Encode waste pile
    num_waste = len(game.waste)
    for card_idx in range(MAX_WASTE_CARDS):
        if card_idx < num_waste:
            card = game.waste[-(MAX_WASTE_CARDS - card_idx)]
            current_index = encode_card(card, card_idx, MAX_WASTE_CARDS, 'waste', 0)
        else:
            current_index += CARD_FEATURES

    return state_vector

def get_state_size() -> int:
    """
    Get the size of the state vector representation.

    Returns:
        The size of the state vector.
    """
    tableau_size = 7 * MAX_TABLEAU_CARDS * CARD_FEATURES
    foundation_size = 4 * MAX_FOUNDATION_CARDS * CARD_FEATURES
    stock_size = 1
    waste_size = MAX_WASTE_CARDS * CARD_FEATURES

    return tableau_size + foundation_size + stock_size + waste_size

def get_state_representation_compact(game: SolitaireGame) -> np.ndarray:
    """
    Alternative compact state representation focusing on key game features.

    This representation includes:
    - Number of cards in each tableau pile
    - Number of cards in each foundation pile
    - Number of cards in stock
    - Number of cards in waste
    - Top card features for each pile (suit, rank, face-up)

    Args:
        game: The SolitaireGame instance to encode.

    Returns:
        A compact numpy array representing the game state.
    """
    # Tableau pile sizes (7 values)
    tableau_sizes = [len(pile) for pile in game.tableau]

    # Foundation pile sizes (4 values)
    foundation_sizes = [len(pile) for pile in game.foundations]

    # Stock and waste sizes (2 values)
    stock_size = len(game.stock.cards)
    waste_size = len(game.waste)

    # Top card features for each pile
    def get_top_card_features(pile):
        """Get features of the top card in a pile."""
        if len(pile) == 0:
            return [0, 0, 0]  # No card: suit=0, rank=0, face_up=0

        top_card = pile[-1]
        suit = ['Hearts', 'Diamonds', 'Clubs', 'Spades'].index(top_card.suit) + 1
        rank = Card.VALID_RANKS.index(top_card.rank) + 1
        face_up = 1 if top_card.face_up else 0

        return [suit, rank, face_up]

    # Tableau top cards (7 piles * 3 features)
    tableau_tops = []
    for pile in game.tableau:
        tableau_tops.extend(get_top_card_features(pile))

    # Foundation top cards (4 piles * 3 features)
    foundation_tops = []
    for pile in game.foundations:
        foundation_tops.extend(get_top_card_features(pile))

    # Waste top cards (up to 3 visible cards * 3 features)
    waste_tops = []
    for i in range(min(3, len(game.waste))):
        card = game.waste[-(i+1)]
        waste_tops.extend(get_top_card_features([card]))

    # Pad waste tops to 3 cards if needed
    while len(waste_tops) < 9:
        waste_tops.append(0)

    # Combine all features
    state_features = (
        tableau_sizes +
        foundation_sizes +
        [stock_size, waste_size] +
        tableau_tops +
        foundation_tops +
        waste_tops
    )

    return np.array(state_features, dtype=np.float32)