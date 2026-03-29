from utils import get_basic_strategy_table
from utils import get_basic_strategy_table
import random
from collections import deque
from typing import List, Tuple, Dict, Any

def hand_value(cards: List[int]) -> int:
    """
    Compute the blackjack hand value, treating aces as 11 where possible without busting.
    """
    val = sum(cards)
    aces = cards.count(11)
    while val > 21 and aces > 0:
        val -= 10
        aces -= 1
    return val

def is_soft_hand(cards: List[int]) -> bool:
    """
    Check if the hand is soft (contains at least one ace counted as 11).
    """
    if 11 not in cards:
        return False
    hard_val = sum(1 if c == 11 else c for c in cards)
    return hand_value(cards) > hard_val

def is_blackjack(cards: List[int]) -> bool:
    """
    Check if the hand is a natural blackjack (two cards totaling 21).
    """
    return len(cards) == 2 and hand_value(cards) == 21

class BasicStrategy:
    """
    Encapsulates precomputed basic strategy lookup tables for perfect play decisions.
    """
    def __init__(self):
        self.table_hard, self.table_soft, self.table_split = get_basic_strategy_table()

    def get_action(self, player_total: int, is_soft: bool, dealer_up: int, can_double: bool, can_split: bool = False) -> str:
        """
        Get basic strategy action: 'H' hit, 'S' stand, 'D' double (else H), 'P' split (if applicable).
        """
        if can_split:
            # Split check integrated if can_split, but typically called post-split check
            pair_val = player_total // 2  # Approximate, but better use separate for pairs
            split_act = self.table_split.get(pair_val, {}).get(dealer_up, 'H')
            if split_act == 'P':
                return 'P'
        table = self.table_soft if is_soft else self.table_hard
        act = table.get(player_total, {}).get(dealer_up, 'H')
        if act == 'D' and not can_double:
            act = 'H'
        return act

    def get_split_action(self, pair_val: int, dealer_up: int) -> str:
        """
        Specific split decision for pair of given value vs dealer upcard.
        """
        return self.table_split.get(pair_val, {}).get(dealer_up, 'H')

class CardCounter:
    """
    Tracks Hi-Lo card counting system: +1 (2-6), 0 (7-9), -1 (10-A).
    """
    def __init__(self):
        self.running_count: int = 0

    def hi_lo_value(self, card: int) -> int:
        if 2 <= card <= 6:
            return 1
        if 7 <= card <= 9:
            return 0
        if 10 <= card <= 11:
            return -1
        return 0

    def update(self, card: int) -> None:
        self.running_count += self.hi_lo_value(card)

    def get_true_count(self, decks_remaining: float) -> float:
        if decks_remaining <= 0:
            return 0.0
        return round(self.running_count / decks_remaining, 2)  # Optional rounding

class BlackjackGame:
    """
    Manages full blackjack shoe, dealing, and hand resolution following rules.
    Returns reward (delta bankroll) for RL, final player hands (for splits), dealer cards.
    """
    def __init__(self, num_decks: int = 6, penetration: float = 0.75):
        self.num_decks: int = num_decks
        self.penetration: float = penetration
        self.total_cards: int = 52 * num_decks
        self.shoe: deque[int] = deque()
        self.new_shoe()

    def new_shoe(self) -> None:
        """Create and shuffle a new shoe with num_decks."""
        single_deck = list(range(2, 12)) * 4
        deck = single_deck * self.num_decks
        random.shuffle(deck)
        self.shoe = deque(deck)

    def deal_card(self) -> int:
        """Deal a card from the shoe, reshuffle if penetration reached or empty."""
        if len(self.shoe) / self.total_cards < (1 - self.penetration) or not self.shoe:
            self.new_shoe()
        return self.shoe.popleft()

    def play_hand(self, initial_bet: float, strategy: BasicStrategy, counter: CardCounter) -> Tuple[float, List[List[int]], List[int]]:
        """
        Simulate full hand: deal, player actions (basic strategy + splits/doubles),
        dealer play, resolution. Updates counter with all cards.
        Returns (delta_bankroll: float, player_final_hands: list[list[int]], dealer_cards: list[int]).
        Handles blackjack early payout.
        Splits up to 4 hands max, doubles on any two cards.
        """
        # Initial deal
        p1 = self.deal_card()
        counter.update(p1)
        p2 = self.deal_card()
        counter.update(p2)
        d_up = self.deal_card()
        counter.update(d_up)

        player_hands: List[List[int]] = [[p1, p2]]
        hand_bets: List[float] = [initial_bet]
        dealer_cards: List[int] = [d_up]

        # Check player blackjack
        if is_blackjack(player_hands[0]):
            d_hole = self.deal_card()
            counter.update(d_hole)
            dealer_cards.append(d_hole)
            if is_blackjack(dealer_cards):
                return 0.0, player_hands, dealer_cards  # Push
            else:
                return 1.5 * initial_bet, player_hands, dealer_cards

        # Handle splits (iterative, max 4 hands)
        i = 0
        while i < len(player_hands):
            cards = player_hands[i]
            hbet = hand_bets[i]
            if (len(cards) == 2 and
                cards[0] == cards[1] and
                len(player_hands) < 4 and
                strategy.get_split_action(cards[0], d_up) == 'P'):
                # Split
                card1 = self.deal_card()
                counter.update(card1)
                card2 = self.deal_card()
                counter.update(card2)
                player_hands[i] = [cards[0], card1]
                player_hands.insert(i + 1, [cards[1], card2])
                hand_bets[i] = hbet
                hand_bets.insert(i + 1, hbet)
                i += 1  # Move to next (newly inserted second hand)
                continue
            i += 1

        # Player plays each hand (hit/double/stand per basic strategy)
        for j in range(len(player_hands)):
            cards = player_hands[j]
            while True:
                total = hand_value(cards)
                is_soft = is_soft_hand(cards)
                can_double = len(cards) == 2
                action = strategy.get_action(total, is_soft, d_up, can_double)
                if action == 'S':
                    break
                new_card = self.deal_card()
                counter.update(new_card)
                cards.append(new_card)
                if action == 'D':
                    hand_bets[j] *= 2
                    break  # Double: one card, then stand

        # Dealer play (stand on soft 17)
        d_hole = self.deal_card()
        counter.update(d_hole)
        dealer_cards.append(d_hole)
        while hand_value(dealer_cards) < 17:
            new_card = self.deal_card()
            counter.update(new_card)
            dealer_cards.append(new_card)

        # Resolve all player hands
        delta: float = 0.0
        d_val = hand_value(dealer_cards)
        d_bust = d_val > 21
        for k in range(len(player_hands)):
            cards = player_hands[k]
            hbet = hand_bets[k]
            p_val = hand_value(cards)
            if p_val > 21:  # Player bust
                delta -= hbet
            elif d_bust or p_val > d_val:
                delta += hbet  # Win: net +bet
            elif p_val == d_val:
                pass  # Push: net 0
            else:  # Loss
                delta -= hbet
        return delta, player_hands, dealer_cards