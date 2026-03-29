"""
action_utils.py - Utility functions for action encoding, decoding, and application.

This module provides functions to work with the action space in the solitaire RL project.
It includes functions to get valid actions, apply actions to the game state, convert between
move tuples and action indices, and calculate rewards according to the game rules.
"""

from typing import List, Tuple
from game.solitaire import SolitaireGame, Card
import config

def get_valid_actions(game: SolitaireGame) -> List[Tuple]:
    """
    Get all valid actions in the current game state.

    This function returns a list of all possible valid moves that can be made
    in the current game state. Each action is represented as a tuple that can
    be directly used with the apply_action function.

    Args:
        game: The current SolitaireGame instance.

    Returns:
        A list of tuples representing valid actions. Each tuple contains:
        - Move type ('to_foundation' or 'to_tableau')
        - Source information (card or list of cards)
        - Destination information (foundation index or tableau index)
    """
    valid_actions = []

    # Check moves from waste to foundations
    if len(game.waste) > 0:
        top_waste = game.waste[-1]
        for i in range(4):
            if game.can_move_to_foundation(top_waste, i):
                valid_actions.append(('to_foundation', top_waste, i))

    # Check moves from waste to tableau
    if len(game.waste) > 0:
        top_waste = game.waste[-1]
        for i in range(7):
            if game.can_move_to_tableau([top_waste], i):
                valid_actions.append(('to_tableau', [top_waste], i))

    # Check moves from tableau to foundations
    for pile_idx, pile in enumerate(game.tableau):
        if len(pile) > 0 and pile[-1].face_up:
            top_card = pile[-1]
            for foundation_idx in range(4):
                if game.can_move_to_foundation(top_card, foundation_idx):
                    valid_actions.append(('to_foundation', top_card, foundation_idx))

    # Check moves from tableau to tableau
    for source_idx, source_pile in enumerate(game.tableau):
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
                if source_idx != target_idx and game.can_move_to_tableau(sequence, target_idx):
                    valid_actions.append(('to_tableau', sequence, target_idx))

    # Add stock deal action if stock has cards or waste has cards to reset
    if not game.stock.is_empty() or len(game.waste) > 0:
        valid_actions.append(('deal_from_stock',))

    return valid_actions

def apply_action(game: SolitaireGame, action: Tuple) -> int:
    """
    Apply an action to the game state and return the reward.

    This function takes an action tuple and applies it to the game state,
    updating the game accordingly. It returns the reward for the action based
    on the game rules.

    Args:
        game: The SolitaireGame instance to modify.
        action: A tuple representing the action to apply. Format:
            - ('to_foundation', card, foundation_idx)
            - ('to_tableau', [cards], tableau_idx)
            - ('deal_from_stock',)

    Returns:
        The reward for the action:
        - +100 for winning the game
        - +10 for moving to foundation
        - +5 for moving to tableau
        - 0 for dealing from stock
        - -1 for invalid moves

    Raises:
        ValueError: If the action format is invalid.
    """
    if not isinstance(action, tuple) or len(action) < 1:
        raise ValueError(f"Invalid action format: {action}. Action must be a non-empty tuple.")

    move_type = action[0]

    if move_type == 'to_foundation':
        if len(action) != 3:
            raise ValueError(f"Invalid foundation move format: {action}. Expected ('to_foundation', card, foundation_idx)")

        card = action[1]
        foundation_idx = action[2]

        if game.move_to_foundation(card, foundation_idx):
            # Check if this move won the game
            if game._check_win():
                return config.REWARD_WIN
            return config.REWARD_FOUNDATION_MOVE
        return config.REWARD_INVALID_MOVE

    elif move_type == 'to_tableau':
        if len(action) != 3:
            raise ValueError(f"Invalid tableau move format: {action}. Expected ('to_tableau', [cards], tableau_idx)")

        cards = action[1]
        tableau_idx = action[2]

        if game.move_to_tableau(cards, tableau_idx):
            return config.REWARD_TABLEAU_MOVE
        return config.REWARD_INVALID_MOVE

    elif move_type == 'deal_from_stock':
        if game.deal_from_stock():
            return 0  # Neutral reward for dealing
        return config.REWARD_INVALID_MOVE

    else:
        raise ValueError(f"Unknown move type: {move_type}")

def action_to_index(action: Tuple, valid_actions: List[Tuple]) -> int:
    """
    Convert an action tuple to its index in the valid actions list.

    This function is used to convert between action tuples and action indices
    for the RL agent. It provides a more robust matching algorithm than simple
    list.index() to handle card object comparisons.

    Args:
        action: The action tuple to convert to an index.
        valid_actions: The current list of valid actions.

    Returns:
        The index of the action in the valid_actions list, or -1 if not found.

    Raises:
        ValueError: If the action tuple format is invalid.
    """
    if not isinstance(action, tuple) or len(action) < 1:
        raise ValueError(f"Invalid action format: {action}")

    move_type = action[0]

    # Special handling for deal_from_stock action
    if move_type == 'deal_from_stock':
        for idx, valid_action in enumerate(valid_actions):
            if valid_action[0] == 'deal_from_stock':
                return idx
        return -1

    # Handle card-based actions
    for idx, valid_action in enumerate(valid_actions):
        if valid_action[0] != move_type:
            continue

        if move_type == 'to_foundation':
            # Compare card and foundation index
            action_card = action[1]
            action_foundation = action[2]
            valid_card = valid_action[1]
            valid_foundation = valid_action[2]

            # Compare cards by their properties (since they might be different objects)
            if (action_card.suit == valid_card.suit and
                action_card.rank == valid_card.rank and
                action_card.face_up == valid_card.face_up and
                action_foundation == valid_foundation):
                return idx

        elif move_type == 'to_tableau':
            # Compare card sequence and tableau index
            action_cards = action[1]
            action_tableau = action[2]
            valid_cards = valid_action[1]
            valid_tableau = valid_action[2]

            # Check tableau index
            if action_tableau != valid_tableau:
                continue

            # Check if sequences match
            if len(action_cards) != len(valid_cards):
                continue

            # Compare each card in the sequence
            cards_match = True
            for action_card, valid_card in zip(action_cards, valid_cards):
                if not (action_card.suit == valid_card.suit and
                       action_card.rank == valid_card.rank and
                       action_card.face_up == valid_card.face_up):
                    cards_match = False
                    break

            if cards_match:
                return idx

    return -1  # Action not found

def index_to_action(index: int, valid_actions: List[Tuple]) -> Tuple:
    """
    Convert an action index to its corresponding action tuple.

    Args:
        index: The index in the valid_actions list.
        valid_actions: The current list of valid actions.

    Returns:
        The action tuple at the specified index.

    Raises:
        IndexError: If the index is out of range.
        TypeError: If index is not an integer.
    """
    if not isinstance(index, int):
        raise TypeError(f"Index must be an integer, got {type(index).__name__}")

    if index < 0 or index >= len(valid_actions):
        raise IndexError(f"Action index {index} out of range for {len(valid_actions)} valid actions")

    return valid_actions[index]

def encode_action_for_neural_network(action: Tuple) -> int:
    """
    Encode an action into a unique integer ID for neural network output.

    This creates a consistent encoding scheme that can be decoded back to
    action parameters. This is useful for having a fixed action space.

    Encoding scheme:
    - Foundation moves: 0-207 (4 foundations × 52 cards)
    - Tableau moves: 208-575 (7 tableaus × 52 cards + 7 tableaus × 1 for sequences)
    - Stock deal: 576

    Args:
        action: The action tuple to encode.

    Returns:
        A unique integer ID for the action.

    Raises:
        ValueError: If the action format is invalid.
    """
    if not isinstance(action, tuple) or len(action) < 1:
        raise ValueError(f"Invalid action format: {action}")

    move_type = action[0]

    if move_type == 'to_foundation':
        if len(action) != 3:
            raise ValueError(f"Invalid foundation move: {action}")

        card = action[1]
        foundation_idx = action[2]

        # Validate foundation index
        if foundation_idx < 0 or foundation_idx > 3:
            raise ValueError(f"Invalid foundation index: {foundation_idx}")

        # Get card ID (0-51)
        suit_id = Card.VALID_SUITS.index(card.suit)
        rank_id = Card.VALID_RANKS.index(card.rank)
        card_id = suit_id * 13 + rank_id

        # Foundation moves: 0-207
        return foundation_idx * 52 + card_id

    elif move_type == 'to_tableau':
        if len(action) != 3:
            raise ValueError(f"Invalid tableau move: {action}")

        cards = action[1]
        tableau_idx = action[2]

        # Validate tableau index
        if tableau_idx < 0 or tableau_idx > 6:
            raise ValueError(f"Invalid tableau index: {tableau_idx}")

        # Get bottom card ID (the card that will be placed)
        bottom_card = cards[0]
        suit_id = Card.VALID_SUITS.index(bottom_card.suit)
        rank_id = Card.VALID_RANKS.index(bottom_card.rank)
        card_id = suit_id * 13 + rank_id

        # Tableau moves: 208-575
        return 208 + tableau_idx * 52 + card_id

    elif move_type == 'deal_from_stock':
        # Stock deal: 576
        return 576

    else:
        raise ValueError(f"Unknown move type: {move_type}")

def decode_action_from_neural_network(action_id: int, game: SolitaireGame) -> Tuple:
    """
    Decode a neural network action ID back to an action tuple.

    This function converts the fixed action space ID back to a specific
    action that can be applied to the game state.

    Args:
        action_id: The encoded action ID.
        game: The current game state (needed to get actual card objects).

    Returns:
        The decoded action tuple, or None if no valid action can be created.

    Raises:
        ValueError: If the action_id is invalid.
    """
    if not isinstance(action_id, int) or action_id < 0:
        raise ValueError(f"Invalid action ID: {action_id}")

    # Stock deal action
    if action_id == 576:
        return ('deal_from_stock',)

    # Tableau moves
    if 208 <= action_id <= 575:
        tableau_id = (action_id - 208) // 52
        card_id = (action_id - 208) % 52

        # Find the actual card in the game state
        suit = Card.VALID_SUITS[card_id // 13]
        rank = Card.VALID_RANKS[card_id % 13]

        # Search for this card in the game state
        for pile in game.tableau:
            for card in pile:
                if card.suit == suit and card.rank == rank and card.face_up:
                    # Found the card, create action
                    # For tableau moves, we typically move just this card
                    return ('to_tableau', [card], tableau_id)

        return None

    # Foundation moves
    if 0 <= action_id <= 207:
        foundation_id = action_id // 52
        card_id = action_id % 52

        # Find the actual card in the game state
        suit = Card.VALID_SUITS[card_id // 13]
        rank = Card.VALID_RANKS[card_id % 13]

        # Search for this card in the game state
        for pile in game.tableau:
            for card in pile:
                if card.suit == suit and card.rank == rank and card.face_up:
                    return ('to_foundation', card, foundation_id)

        # Check waste pile
        if len(game.waste) > 0:
            top_waste = game.waste[-1]
            if top_waste.suit == suit and top_waste.rank == rank:
                return ('to_foundation', top_waste, foundation_id)

        return None

    raise ValueError(f"Invalid action ID: {action_id}")

def get_max_action_id() -> int:
    """
    Get the maximum valid action ID for neural network output size.

    Returns:
        The maximum action ID (576 for the current encoding scheme).
    """
    return 576  # Foundation(208) + Tableau(368) + Stock(1) = 577 total actions