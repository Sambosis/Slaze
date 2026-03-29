"""
test_state_encoding.py - Unit tests for the state encoding functionality.

This module contains comprehensive unit tests for the state encoder functions,
ensuring consistent and correct vectorization of game states for neural network input.
It validates the encoding process and checks for edge cases.
"""

import numpy as np
import pytest
from utils.state_encoder import get_state_representation, get_state_size, get_state_representation_compact
from game.solitaire import SolitaireGame, Card, Deck

def test_state_size_consistency():
    """Test that the state size is consistent across different game instances."""
    game1 = SolitaireGame()
    game2 = SolitaireGame()

    state1 = get_state_representation(game1)
    state2 = get_state_representation(game2)

    # Both states should have the same size
    assert state1.shape == state2.shape
    assert len(state1) == get_state_size()

def test_empty_game_state():
    """Test state representation of a game with empty piles."""
    game = SolitaireGame()

    # Clear all piles for testing
    game.tableau = [[] for _ in range(7)]
    game.foundations = [[] for _ in range(4)]
    game.stock.cards = []
    game.waste = []

    state = get_state_representation(game)

    # State should be all zeros for empty game
    assert np.allclose(state, np.zeros_like(state))

def test_single_card_in_tableau():
    """Test state representation with a single card in tableau."""
    game = SolitaireGame()

    # Clear all piles
    game.tableau = [[] for _ in range(7)]
    game.foundations = [[] for _ in range(4)]
    game.stock.cards = []
    game.waste = []

    # Add a single card to first tableau pile
    card = Card('Hearts', 'Ace', face_up=True)
    game.tableau[0].append(card)

    state = get_state_representation(game)

    # Check that the state is not all zeros
    assert not np.allclose(state, np.zeros_like(state))

def test_card_features_encoding():
    """Test that card features are properly encoded."""
    game = SolitaireGame()

    # Clear all piles
    game.tableau = [[] for _ in range(7)]
    game.foundations = [[] for _ in range(4)]
    game.stock.cards = []
    game.waste = []

    # Add specific cards to test feature encoding
    # Card 1: Hearts Ace (face up)
    card1 = Card('Hearts', 'Ace', face_up=True)
    game.tableau[0].append(card1)

    # Card 2: Spades King (face down)
    card2 = Card('Spades', 'King', face_up=False)
    game.tableau[1].append(card2)

    state = get_state_representation(game)

    # The state should contain non-zero values for these cards
    assert not np.allclose(state, np.zeros_like(state))

def test_foundation_cards():
    """Test state representation with cards in foundation piles."""
    game = SolitaireGame()

    # Clear all piles
    game.tableau = [[] for _ in range(7)]
    game.foundations = [[] for _ in range(4)]
    game.stock.cards = []
    game.waste = []

    # Add cards to foundations
    # Foundation 0: Hearts Ace
    card1 = Card('Hearts', 'Ace', face_up=True)
    game.foundations[0].append(card1)

    # Foundation 1: Diamonds 2
    card2 = Card('Diamonds', '2', face_up=True)
    game.foundations[1].append(card2)

    state = get_state_representation(game)

    # State should contain foundation card information
    assert not np.allclose(state, np.zeros_like(state))

def test_waste_pile():
    """Test state representation with cards in waste pile."""
    game = SolitaireGame()

    # Clear all piles
    game.tableau = [[] for _ in range(7)]
    game.foundations = [[] for _ in range(4)]
    game.stock.cards = []
    game.waste = []

    # Add cards to waste pile
    card1 = Card('Clubs', '5', face_up=True)
    card2 = Card('Diamonds', 'Queen', face_up=True)
    game.waste.extend([card1, card2])

    state = get_state_representation(game)

    # State should contain waste pile information
    assert not np.allclose(state, np.zeros_like(state))

def test_stock_count():
    """Test that stock count is properly encoded."""
    game = SolitaireGame()

    # Clear all piles
    game.tableau = [[] for _ in range(7)]
    game.foundations = [[] for _ in range(4)]
    game.stock.cards = []
    game.waste = []

    # Add specific number of cards to stock
    for i in range(10):
        game.stock.cards.append(Card('Hearts', str((i % 13) + 1), face_up=False))

    state = get_state_representation(game)

    # The stock count should be encoded
    assert not np.allclose(state, np.zeros_like(state))

def test_compact_state_representation():
    """Test the compact state representation function."""
    game = SolitaireGame()

    # Get both representations
    full_state = get_state_representation(game)
    compact_state = get_state_representation_compact(game)

    # Compact state should be smaller than full state
    assert len(compact_state) < len(full_state)

    # Compact state should not be all zeros
    assert not np.allclose(compact_state, np.zeros_like(compact_state))

def test_state_consistency_across_games():
    """Test that similar game states produce similar representations."""
    # Create two games with similar states
    game1 = SolitaireGame()
    game2 = SolitaireGame()

    # Clear both games
    game1.tableau = [[] for _ in range(7)]
    game1.foundations = [[] for _ in range(4)]
    game1.stock.cards = []
    game1.waste = []

    game2.tableau = [[] for _ in range(7)]
    game2.foundations = [[] for _ in range(4)]
    game2.stock.cards = []
    game2.waste = []

    # Add the same card to both games
    card1 = Card('Hearts', 'Ace', face_up=True)
    card2 = Card('Hearts', 'Ace', face_up=True)

    game1.tableau[0].append(card1)
    game2.tableau[0].append(card2)

    state1 = get_state_representation(game1)
    state2 = get_state_representation(game2)

    # States should be identical for identical game states
    assert np.allclose(state1, state2)

def test_different_card_positions():
    """Test that cards in different positions produce different states."""
    game1 = SolitaireGame()
    game2 = SolitaireGame()

    # Clear both games
    game1.tableau = [[] for _ in range(7)]
    game1.foundations = [[] for _ in range(4)]
    game1.stock.cards = []
    game1.waste = []

    game2.tableau = [[] for _ in range(7)]
    game2.foundations = [[] for _ in range(4)]
    game2.stock.cards = []
    game2.waste = []

    # Add same card to different tableau piles
    card1 = Card('Hearts', 'Ace', face_up=True)
    card2 = Card('Hearts', 'Ace', face_up=True)

    game1.tableau[0].append(card1)  # First pile
    game2.tableau[1].append(card2)  # Second pile

    state1 = get_state_representation(game1)
    state2 = get_state_representation(game2)

    # States should be different due to different positions
    assert not np.allclose(state1, state2)

def test_face_up_vs_face_down():
    """Test that face-up and face-down cards are encoded differently."""
    game1 = SolitaireGame()
    game2 = SolitaireGame()

    # Clear both games
    game1.tableau = [[] for _ in range(7)]
    game1.foundations = [[] for _ in range(4)]
    game1.stock.cards = []
    game1.waste = []

    game2.tableau = [[] for _ in range(7)]
    game2.foundations = [[] for _ in range(4)]
    game2.stock.cards = []
    game2.waste = []

    # Add same card but with different face-up status
    card1 = Card('Hearts', 'Ace', face_up=True)
    card2 = Card('Hearts', 'Ace', face_up=False)

    game1.tableau[0].append(card1)  # Face up
    game2.tableau[0].append(card2)  # Face down

    state1 = get_state_representation(game1)
    state2 = get_state_representation(game2)

    # States should be different due to face-up status
    assert not np.allclose(state1, state2)

def test_state_vector_type_and_shape():
    """Test that the state vector has the correct type and shape."""
    game = SolitaireGame()
    state = get_state_representation(game)

    # Should be a numpy array
    assert isinstance(state, np.ndarray)

    # Should be float32 type
    assert state.dtype == np.float32

    # Should be 1D vector
    assert len(state.shape) == 1

    # Should have the expected size
    expected_size = get_state_size()
    assert len(state) == expected_size

def test_compact_state_features():
    """Test that compact state contains expected features."""
    game = SolitaireGame()

    # Clear all piles
    game.tableau = [[] for _ in range(7)]
    game.foundations = [[] for _ in range(4)]
    game.stock.cards = []
    game.waste = []

    # Add specific cards to create predictable state
    # Tableau: Add 1 card to first pile
    card1 = Card('Hearts', 'Ace', face_up=True)
    game.tableau[0].append(card1)

    # Foundation: Add 1 card to first foundation
    card2 = Card('Hearts', 'Ace', face_up=True)
    game.foundations[0].append(card2)

    # Waste: Add 1 card
    card3 = Card('Clubs', '5', face_up=True)
    game.waste.append(card3)

    # Stock: Add 5 cards
    for i in range(5):
        game.stock.cards.append(Card('Diamonds', str((i % 13) + 1), face_up=False))

    compact_state = get_state_representation_compact(game)

    # Compact state should contain:
    # - 7 tableau sizes
    # - 4 foundation sizes
    # - 2 stock/waste sizes
    # - 7 * 3 tableau top card features
    # - 4 * 3 foundation top card features
    # - 3 * 3 waste top card features
    expected_length = 7 + 4 + 2 + 7*3 + 4*3 + 3*3
    assert len(compact_state) == expected_length

    # Check that the state is not all zeros
    assert not np.allclose(compact_state, np.zeros_like(compact_state))

def test_edge_case_full_tableau():
    """Test state representation with maximum cards in tableau piles."""
    game = SolitaireGame()

    # Clear all piles
    game.tableau = [[] for _ in range(7)]
    game.foundations = [[] for _ in range(4)]
    game.stock.cards = []
    game.waste = []

    # Fill first tableau pile with maximum cards
    for i in range(19):  # MAX_TABLEAU_CARDS = 19
        card = Card('Hearts', str((i % 13) + 1), face_up=True)
        game.tableau[0].append(card)

    state = get_state_representation(game)

    # State should handle the full pile correctly
    assert len(state) == get_state_size()
    assert not np.allclose(state, np.zeros_like(state))

def test_edge_case_empty_after_full():
    """Test state representation after clearing a full pile."""
    game = SolitaireGame()

    # Clear all piles
    game.tableau = [[] for _ in range(7)]
    game.foundations = [[] for _ in range(4)]
    game.stock.cards = []
    game.waste = []

    # Fill and then empty a tableau pile
    for i in range(10):
        card = Card('Hearts', str((i % 13) + 1), face_up=True)
        game.tableau[0].append(card)

    # Now empty it
    game.tableau[0] = []

    state = get_state_representation(game)

    # State should handle the empty pile correctly
    assert len(state) == get_state_size()

def test_state_representation_deterministic():
    """Test that state representation is deterministic for the same game state."""
    game = SolitaireGame()

    # Get state representation multiple times
    state1 = get_state_representation(game)
    state2 = get_state_representation(game)
    state3 = get_state_representation(game)

    # All representations should be identical
    assert np.allclose(state1, state2)
    assert np.allclose(state2, state3)
    assert np.allclose(state1, state3)

def test_different_suits_and_ranks():
    """Test that different suits and ranks are encoded differently."""
    game1 = SolitaireGame()
    game2 = SolitaireGame()
    game3 = SolitaireGame()

    # Clear all games
    for game in [game1, game2, game3]:
        game.tableau = [[] for _ in range(7)]
        game.foundations = [[] for _ in range(4)]
        game.stock.cards = []
        game.waste = []

    # Different suits
    card1 = Card('Hearts', 'Ace', face_up=True)
    card2 = Card('Diamonds', 'Ace', face_up=True)

    game1.tableau[0].append(card1)
    game2.tableau[0].append(card2)

    # Different ranks
    card3 = Card('Hearts', 'King', face_up=True)
    game3.tableau[0].append(card3)

    state1 = get_state_representation(game1)
    state2 = get_state_representation(game2)
    state3 = get_state_representation(game3)

    # Different suits should produce different states
    assert not np.allclose(state1, state2)

    # Different ranks should produce different states
    assert not np.allclose(state1, state3)

def test_state_normalization():
    """Test that state values are properly normalized."""
    game = SolitaireGame()
    state = get_state_representation(game)

    # Most values should be normalized (0 or 1 for binary features,
    # 0-1 for continuous features like position)
    assert state.min() >= 0, "No values should be negative"
    assert state.max() <= 1.5, "Values should not be excessively large"

    # Check for reasonable distribution (should have both 0s and 1s)
    unique_values = np.unique(state)
    assert len(unique_values) > 1, "State should have some variation"

def test_integration_with_neural_network():
    """Test that the state encoding can be used with neural networks."""
    try:
        import torch

        # Create a game
        game = SolitaireGame()
        state = get_state_representation(game)

        # Convert to tensor (this should work without errors)
        state_tensor = torch.FloatTensor(state)

        # Verify tensor properties
        assert state_tensor.shape[0] == len(state), "Tensor should preserve state size"
        assert state_tensor.dtype == torch.float32, "Tensor should be float32"

    except ImportError:
        # PyTorch not available, test with numpy only
        game = SolitaireGame()
        state = get_state_representation(game)

        # State should be suitable for neural network input
        assert state.shape == (get_state_size(),), "State should have expected shape"
        assert state.dtype == np.float32, "State should be float32"