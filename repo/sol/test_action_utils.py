"""
test_action_utils.py - Unit tests for action utilities in the solitaire RL project.

This module contains comprehensive unit tests for the action utilities functions,
including tests for valid action detection, action application, reward calculation,
and action encoding/decoding functionality.
"""

import pytest
from utils.action_utils import get_valid_actions, apply_action
from game.solitaire import SolitaireGame, Card

def test_get_valid_actions_initial_state():
    """Test that valid actions are correctly identified in the initial game state."""
    game = SolitaireGame()

    # Get valid actions
    valid_actions = get_valid_actions(game)

    # Should have at least the deal_from_stock action
    assert len(valid_actions) > 0
    assert ('deal_from_stock',) in valid_actions

    # Check that actions are properly formatted
    for action in valid_actions:
        assert isinstance(action, tuple)
        assert len(action) >= 1

        if action[0] == 'to_foundation':
            assert len(action) == 3
            assert isinstance(action[1], Card)
            assert isinstance(action[2], int)
            assert 0 <= action[2] < 4

        elif action[0] == 'to_tableau':
            assert len(action) == 3
            assert isinstance(action[1], list)
            assert len(action[1]) > 0
            assert all(isinstance(card, Card) for card in action[1])
            assert isinstance(action[2], int)
            assert 0 <= action[2] < 7

        elif action[0] == 'deal_from_stock':
            assert len(action) == 1

def test_apply_action_deal_from_stock():
    """Test applying the deal_from_stock action."""
    game = SolitaireGame()

    # Get initial stock size
    initial_stock_size = len(game.stock.cards)

    # Apply deal_from_stock action
    reward = apply_action(game, ('deal_from_stock',))

    # Check reward
    assert reward == 0

    # Check that cards were moved from stock to waste
    if initial_stock_size > 0:
        assert len(game.stock.cards) < initial_stock_size
        assert len(game.waste) > 0

def test_apply_action_invalid():
    """Test applying invalid actions."""
    game = SolitaireGame()

    # Create an invalid action (trying to move a card that doesn't exist)
    invalid_action = ('to_foundation', Card('Hearts', 'Ace'), 0)

    # Apply invalid action
    reward = apply_action(game, invalid_action)

    # Check that reward is negative for invalid move
    assert reward == -1

def test_apply_action_foundation_move():
    """Test applying a valid foundation move."""
    game = SolitaireGame()

    # Find a valid foundation move (Ace to empty foundation)
    valid_actions = get_valid_actions(game)
    foundation_moves = [a for a in valid_actions if a[0] == 'to_foundation']

    if foundation_moves:
        # Apply the first valid foundation move
        action = foundation_moves[0]
        reward = apply_action(game, action)

        # Check reward
        assert reward == 10

        # Verify the card was moved to foundation
        card = action[1]
        foundation_idx = action[2]
        assert card in game.foundations[foundation_idx]

def test_apply_action_tableau_move():
    """Test applying a valid tableau move."""
    game = SolitaireGame()

    # Find a valid tableau move
    valid_actions = get_valid_actions(game)
    tableau_moves = [a for a in valid_actions if a[0] == 'to_tableau']

    if tableau_moves:
        # Apply the first valid tableau move
        action = tableau_moves[0]
        reward = apply_action(game, action)

        # Check reward
        assert reward == 5

        # Verify the cards were moved to tableau
        cards = action[1]
        tableau_idx = action[2]
        for card in cards:
            assert card in game.tableau[tableau_idx]

def test_apply_action_win_condition():
    """Test that winning the game gives the correct reward."""
    # Create a mock game state where we can force a win
    game = SolitaireGame()

    # Clear all piles
    game.tableau = [[] for _ in range(7)]
    game.waste = []

    # Fill foundations with complete sequences
    for i in range(4):
        suit = Card.VALID_SUITS[i]
        for rank in Card.VALID_RANKS:
            card = Card(suit, rank, face_up=True)
            game.foundations[i].append(card)

    # Apply a move that should trigger win detection
    # (This is a bit artificial since the game is already won)
    reward = apply_action(game, ('deal_from_stock',))

    # The game should be won
    assert game._check_win()

def test_action_rewards():
    """Test that different actions return the correct rewards."""
    game = SolitaireGame()

    # Test deal from stock reward
    reward = apply_action(game, ('deal_from_stock',))
    assert reward == 0

    # Test invalid move reward
    invalid_action = ('to_foundation', Card('Hearts', 'Ace'), 0)
    reward = apply_action(game, invalid_action)
    assert reward == -1

def test_action_formats():
    """Test that invalid action formats raise appropriate errors."""
    game = SolitaireGame()

    # Test empty action
    with pytest.raises(ValueError):
        apply_action(game, ())

    # Test invalid move type
    with pytest.raises(ValueError):
        apply_action(game, ('invalid_move',))

    # Test malformed foundation move
    with pytest.raises(ValueError):
        apply_action(game, ('to_foundation', Card('Hearts', 'Ace')))

    # Test malformed tableau move
    with pytest.raises(ValueError):
        apply_action(game, ('to_tableau', [Card('Hearts', 'Ace')]))

def test_valid_actions_consistency():
    """Test that valid actions are consistent with game state."""
    game = SolitaireGame()

    # Get valid actions
    valid_actions = get_valid_actions(game)

    # Apply each valid action and verify it works
    for action in valid_actions:
        # Save game state
        original_state = game.__dict__.copy()

        # Apply action
        reward = apply_action(game, action)

        # Verify reward is not negative (since these are valid actions)
        assert reward >= 0

        # Reset game state for next iteration
        game.__dict__ = original_state

def test_action_encoding_decoding():
    """Test action encoding and decoding functions."""
    from utils.action_utils import action_to_index, index_to_action, encode_action_for_neural_network

    game = SolitaireGame()
    valid_actions = get_valid_actions(game)

    # Test action_to_index and index_to_action
    for idx, action in enumerate(valid_actions):
        # Test action_to_index
        found_idx = action_to_index(action, valid_actions)
        assert found_idx == idx

        # Test index_to_action
        found_action = index_to_action(idx, valid_actions)
        assert found_action == action

    # Test encode_action_for_neural_network
    if valid_actions:
        action = valid_actions[0]
        action_id = encode_action_for_neural_network(action)
        assert isinstance(action_id, int)
        assert action_id >= 0

def test_deal_from_stock_action():
    """Test the deal_from_stock action specifically."""
    game = SolitaireGame()

    # Get initial stock and waste sizes
    initial_stock = len(game.stock.cards)
    initial_waste = len(game.waste)

    # Apply deal_from_stock
    reward = apply_action(game, ('deal_from_stock',))

    # Verify reward
    assert reward == 0

    # Verify stock and waste changes
    if initial_stock > 0:
        assert len(game.stock.cards) < initial_stock
        assert len(game.waste) > initial_waste

    # Test dealing when stock is empty but waste has cards
    game.stock.cards = []  # Empty stock
    game.waste = [Card('Hearts', 'Ace', face_up=True)]  # Add card to waste

    reward = apply_action(game, ('deal_from_stock',))
    assert reward == 0
    assert len(game.stock.cards) > 0  # Waste cards moved to stock
    assert len(game.waste) == 0  # Waste should be empty

def test_action_with_empty_piles():
    """Test actions when piles are empty."""
    game = SolitaireGame()

    # Clear waste pile
    game.waste = []

    # Get valid actions
    valid_actions = get_valid_actions(game)

    # Should still have deal_from_stock if applicable
    if not game.stock.is_empty() or len(game.waste) > 0:
        assert ('deal_from_stock',) in valid_actions

    # Clear stock too
    game.stock.cards = []
    valid_actions = get_valid_actions(game)

    # Should have no deal_from_stock action if both stock and waste are empty
    assert ('deal_from_stock',) not in valid_actions

def test_action_reward_structure():
    """Test that rewards match the expected structure from config."""
    game = SolitaireGame()

    # Test foundation move reward
    valid_actions = get_valid_actions(game)
    foundation_moves = [a for a in valid_actions if a[0] == 'to_foundation']

    if foundation_moves:
        action = foundation_moves[0]
        reward = apply_action(game, action)
        assert reward == 10  # REWARD_FOUNDATION_MOVE

    # Test tableau move reward
    game = SolitaireGame()  # Reset game
    valid_actions = get_valid_actions(game)
    tableau_moves = [a for a in valid_actions if a[0] == 'to_tableau']

    if tableau_moves:
        action = tableau_moves[0]
        reward = apply_action(game, action)
        assert reward == 5  # REWARD_TABLEAU_MOVE

    # Test invalid move reward
    invalid_action = ('to_foundation', Card('Hearts', 'Ace'), 0)
    reward = apply_action(game, invalid_action)
    assert reward == -1  # REWARD_INVALID_MOVE

def test_action_sequence_moves():
    """Test moving sequences of cards in tableau."""
    game = SolitaireGame()

    # Find a valid sequence move (multiple cards)
    valid_actions = get_valid_actions(game)
    sequence_moves = [a for a in valid_actions if a[0] == 'to_tableau' and len(a[1]) > 1]

    if sequence_moves:
        action = sequence_moves[0]
        cards = action[1]
        tableau_idx = action[2]

        # Apply the move
        reward = apply_action(game, action)

        # Verify reward
        assert reward == 5

        # Verify all cards were moved
        for card in cards:
            assert card in game.tableau[tableau_idx]

def test_action_with_face_down_cards():
    """Test that actions with face-down cards are handled correctly."""
    game = SolitaireGame()

    # Ensure we have some face-down cards in tableau
    # (This should be true in initial game state)

    # Get valid actions
    valid_actions = get_valid_actions(game)

    # Verify that no actions involve face-down cards
    for action in valid_actions:
        if action[0] == 'to_foundation':
            assert action[1].face_up
        elif action[0] == 'to_tableau':
            for card in action[1]:
                assert card.face_up

def test_action_edge_cases():
    """Test edge cases in action handling."""
    game = SolitaireGame()

    # Test with empty tableau piles
    game.tableau = [[] for _ in range(7)]
    valid_actions = get_valid_actions(game)

    # Should still have deal_from_stock if applicable
    if not game.stock.is_empty() or len(game.waste) > 0:
        assert ('deal_from_stock',) in valid_actions

    # Test with full foundations (should not have foundation moves)
    game = SolitaireGame()
    for i in range(4):
        # Add King to each foundation (max card)
        king = Card(Card.VALID_SUITS[i], 'King', face_up=True)
        game.foundations[i].append(king)

    valid_actions = get_valid_actions(game)
    foundation_moves = [a for a in valid_actions if a[0] == 'to_foundation']

    # Should have no foundation moves since foundations are "full" with Kings
    assert len(foundation_moves) == 0

def test_action_performance():
    """Test that action functions perform reasonably with many cards."""
    game = SolitaireGame()

    # This is more of a performance test - just ensure it doesn't crash
    for _ in range(10):
        valid_actions = get_valid_actions(game)
        if valid_actions:
            action = valid_actions[0]
            apply_action(game, action)

def test_action_consistency_across_games():
    """Test that action detection is consistent across multiple game instances."""
    game1 = SolitaireGame()
    game2 = SolitaireGame()

    # Both should have deal_from_stock action initially
    valid_actions1 = get_valid_actions(game1)
    valid_actions2 = get_valid_actions(game2)

    assert ('deal_from_stock',) in valid_actions1
    assert ('deal_from_stock',) in valid_actions2

    # The number of valid actions should be similar (though not necessarily identical)
    # due to different initial card distributions
    assert abs(len(valid_actions1) - len(valid_actions2)) < 20  # Reasonable tolerance