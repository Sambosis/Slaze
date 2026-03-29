"""
Core blackjack simulation logic for BlackjackBetOptimizer.
Contains Card, Shoe, Hand, BasicStrategy, and Game classes.
"""

import random
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass(frozen=True)
class Card:
    """Represents a single playing card with blackjack value (10JQK=10, A=11)."""
    rank: int  # 2-11 (11=A, 10=10/J/Q/K)
    suit: str  # 'H', 'D', 'C', 'S'
    value: int  # 2-9=face, 10=10/J/Q/K, 11=A

    def __str__(self) -> str:
        rank_map = {11: 'A', 10: '10', 9: '9', 8: '8', 7: '7', 6: '6', 5: '5', 4: '4', 3: '3', 2: '2'}
        suit_map = {'H': '♥', 'D': '♦', 'C': '♣', 'S': '♠'}
        return f"{rank_map[self.rank]}{suit_map.get(self.suit, '?')}"


class Shoe:
    """Manages multi-deck shoe, shuffling, dealing cards, updating Hi-Lo running count."""

    def __init__(self, decks: int):
        self.decks: int = decks
        self.cards: List[Card] = []
        self.running_count: int = 0
        self.cards_dealt: int = 0
        self.shuffle()

    def shuffle(self) -> None:
        """Create and shuffle a fresh shoe, reset count and dealt."""
        suits = ['H', 'D', 'C', 'S']
        ranks = [2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11]
        self.cards = [
            Card(r, s, 11 if r == 11 else min(10, r))
            for _ in range(self.decks)
            for s in suits
            for r in ranks
        ]
        random.shuffle(self.cards)
        self.running_count = 0
        self.cards_dealt = 0

    def deal_card(self) -> Optional[Card]:
        """Deal top card from shoe (pop end for efficiency), update count and dealt."""
        if not self.cards:
            return None
        card = self.cards.pop()
        self.update_count(card)
        self.cards_dealt += 1
        return card

    def update_count(self, card: Card) -> None:
        """Update Hi-Lo running count: 2-6 +1, 10-A -1, 7-9 0."""
        if 2 <= card.rank <= 6:
            self.running_count += 1
        elif card.rank >= 10:
            self.running_count -= 1

    def get_remaining_decks(self) -> float:
        """Estimate remaining decks."""
        return len(self.cards) / 52.0

    def get_true_count(self) -> float:
        """True count = running count / remaining decks."""
        rem_decks = self.get_remaining_decks()
        return self.running_count / rem_decks if rem_decks > 0 else 0.0

    def reshuffle_if_needed(self, penetration: float) -> bool:
        """Reshuffle if dealt fraction >= penetration."""
        dealt_frac = self.cards_dealt / (self.decks * 52)
        if dealt_frac >= penetration:
            self.shuffle()
            return True
        return False


class Hand:
    """Tracks a player or dealer hand, computes totals (soft/hard), checks bust/BJ."""

    def __init__(self, bet: float = 0.0, is_split: bool = False):
        self.cards: List[Card] = []
        self.bet: float = bet
        self.is_split: bool = is_split
        self._total: int = 0
        self._is_soft: bool = False

    def add_card(self, card: Card) -> None:
        """Add card to hand and recalculate total/soft."""
        self.cards.append(card)
        self._update_total()

    def _update_total(self) -> None:
        """Compute total handling aces (11/1), update soft status."""
        if not self.cards:
            self._total = 0
            self._is_soft = False
            return
        total = sum(card.value for card in self.cards)
        num_aces = sum(1 for card in self.cards if card.rank == 11)
        while total > 21 and num_aces > 0:
            total -= 10
            num_aces -= 1
        self._total = total
        self._is_soft = num_aces > 0

    @property
    def total(self) -> int:
        """Current hand total (<=21 with optimal ace valuation)."""
        if self.cards and self._total == 0:
            self._update_total()
        return self._total

    @property
    def is_soft(self) -> bool:
        """True if hand has usable ace (counted as 11)."""
        if self.cards and self._is_soft == False:  # Avoid unnecessary update if already set
            self._update_total()
        return self._is_soft

    def is_bust(self) -> bool:
        """True if total > 21."""
        return self.total > 21

    def is_blackjack(self) -> bool:
        """True if initial two cards form natural blackjack (total 21)."""
        return len(self.cards) == 2 and self.total == 21


class BasicStrategy:
    """Provides lookup for basic strategy actions: 'H', 'S', 'D', 'P'."""

    def __init__(self):
        self.hard_table: Dict[int, Dict[int, str]] = self._init_hard_table()
        self.soft_table: Dict[int, Dict[int, str]] = self._init_soft_table()
        self.pair_table: Dict[int, Dict[int, str]] = self._init_pair_table()

    def _init_hard_table(self) -> Dict[int, Dict[int, str]]:
        table = {
            4: {d: 'H' for d in range(2, 12)},
            5: {d: 'H' for d in range(2, 12)},
            6: {d: 'H' for d in range(2, 12)},
            7: {d: 'H' for d in range(2, 12)},
            8: {d: 'H' for d in range(2, 12)},
            9: {2: 'H', 3: 'D', 4: 'D', 5: 'D', 6: 'D', 7: 'H', 8: 'H', 9: 'H', 10: 'H', 11: 'H'},
            10: {2: 'D', 3: 'D', 4: 'D', 5: 'D', 6: 'D', 7: 'D', 8: 'D', 9: 'D', 10: 'H', 11: 'H'},
            11: {2: 'D', 3: 'D', 4: 'D', 5: 'D', 6: 'D', 7: 'D', 8: 'D', 9: 'D', 10: 'D', 11: 'H'},
            12: {2: 'H', 3: 'H', 4: 'S', 5: 'S', 6: 'S', 7: 'H', 8: 'H', 9: 'H', 10: 'H', 11: 'H'},
            13: {2: 'S', 3: 'S', 4: 'S', 5: 'S', 6: 'S', 7: 'H', 8: 'H', 9: 'H', 10: 'H', 11: 'H'},
            14: {2: 'S', 3: 'S', 4: 'S', 5: 'S', 6: 'S', 7: 'H', 8: 'H', 9: 'H', 10: 'H', 11: 'H'},
            15: {2: 'S', 3: 'S', 4: 'S', 5: 'S', 6: 'S', 7: 'H', 8: 'H', 9: 'H', 10: 'H', 11: 'H'},
            16: {2: 'S', 3: 'S', 4: 'S', 5: 'S', 6: 'S', 7: 'H', 8: 'H', 9: 'H', 10: 'H', 11: 'H'},
        }
        for t in range(17, 22):
            table[t] = {d: 'S' for d in range(2, 12)}
        return table

    def _init_soft_table(self) -> Dict[int, Dict[int, str]]:
        return {
            13: {2: 'H', 3: 'H', 4: 'H', 5: 'D', 6: 'D', 7: 'H', 8: 'H', 9: 'H', 10: 'H', 11: 'H'},
            14: {2: 'H', 3: 'H', 4: 'D', 5: 'D', 6: 'D', 7: 'H', 8: 'H', 9: 'H', 10: 'H', 11: 'H'},
            15: {2: 'H', 3: 'H', 4: 'D', 5: 'D', 6: 'D', 7: 'H', 8: 'H', 9: 'H', 10: 'H', 11: 'H'},
            16: {2: 'H', 3: 'D', 4: 'D', 5: 'D', 6: 'D', 7: 'H', 8: 'H', 9: 'H', 10: 'H', 11: 'H'},
            17: {2: 'H', 3: 'D', 4: 'D', 5: 'D', 6: 'D', 7: 'H', 8: 'H', 9: 'H', 10: 'H', 11: 'H'},
            18: {2: 'S', 3: 'D', 4: 'D', 5: 'D', 6: 'D', 7: 'S', 8: 'S', 9: 'H', 10: 'H', 11: 'H'},
            19: {d: 'S' for d in range(2, 12)},
            20: {d: 'S' for d in range(2, 12)},
        }

    def _init_pair_table(self) -> Dict[int, Dict[int, str]]:
        return {
            2: {2: 'P', 3: 'P', 4: 'P', 5: 'P', 6: 'P', 7: 'P', 8: 'H', 9: 'H', 10: 'H', 11: 'H'},
            3: {2: 'P', 3: 'P', 4: 'P', 5: 'P', 6: 'P', 7: 'P', 8: 'H', 9: 'H', 10: 'H', 11: 'H'},
            4: {2: 'H', 3: 'H', 4: 'H', 5: 'P', 6: 'P', 7: 'H', 8: 'H', 9: 'H', 10: 'H', 11: 'H'},
            5: {2: 'D', 3: 'D', 4: 'D', 5: 'D', 6: 'D', 7: 'D', 8: 'H', 9: 'H', 10: 'H', 11: 'H'},
            6: {2: 'P', 3: 'P', 4: 'P', 5: 'P', 6: 'P', 7: 'H', 8: 'H', 9: 'H', 10: 'H', 11: 'H'},
            7: {2: 'P', 3: 'P', 4: 'P', 5: 'P', 6: 'P', 7: 'P', 8: 'H', 9: 'H', 10: 'H', 11: 'H'},
            8: {d: 'P' for d in range(2, 12)},
            9: {2: 'P', 3: 'P', 4: 'P', 5: 'P', 6: 'P', 7: 'S', 8: 'P', 9: 'P', 10: 'H', 11: 'H'},
            10: {d: 'S' for d in range(2, 12)},
            11: {d: 'P' for d in range(2, 12)},
        }

    def get_action(self, player_total: int, dealer_up: int, usable_ace: bool) -> str:
        """Get basic strategy action for non-pair hands."""
        table = self.soft_table if usable_ace else self.hard_table
        row = table.get(player_total, {d: 'S' for d in range(2, 12)})
        return row.get(dealer_up, 'S')

    def get_pair_action(self, pair_rank: int, dealer_up: int) -> str:
        """Get basic strategy action for pairs."""
        row = self.pair_table.get(pair_rank, {d: 'H' for d in range(2, 12)})
        return row.get(dealer_up, 'H')


class Game:
    """Simulates one full hand: deal, basic strategy decisions, dealer play, resolve payouts."""

    def __init__(self, config: dict):
        self.config = config
        self.shoe = Shoe(config['num_decks'])
        self.player_hands: List[Hand] = []
        self.dealer_hand = Hand()
        self.bankroll: float = config['initial_bankroll']
        self.basic_strategy = BasicStrategy()

    def reset(self) -> None:
        """Reset hands for new hand."""
        self.player_hands = []
        self.dealer_hand = Hand()

    def _resolve_hand(self, player_hand: Hand) -> float:
        """Resolve single player hand vs dealer. Returns payout for that hand."""
        if player_hand.is_bust():
            return -player_hand.bet
        if self.dealer_hand.is_bust():
            return player_hand.bet
        if player_hand.total > self.dealer_hand.total:
            return player_hand.bet
        if player_hand.total < self.dealer_hand.total:
            return -player_hand.bet
        return 0.0  # push

    def _play_player_hands(self) -> None:
        """Player turn: basic strategy actions, handle doubles/splits (up to 4 hands)."""
        i = 0
        max_hands = 4
        while i < len(self.player_hands):
            hand = self.player_hands[i]
            if hand.is_bust():
                i += 1
                continue
            # Stand on split aces (one card each)
            if (hand.is_split and len(hand.cards) == 2 and
                any(card.rank == 11 for card in hand.cards)):
                i += 1
                continue
            dealer_up = self.dealer_hand.cards[0].rank
            is_pair = (len(hand.cards) == 2 and
                       hand.cards[0].rank == hand.cards[1].rank)
            if is_pair:
                action = self.basic_strategy.get_pair_action(hand.cards[0].rank, dealer_up)
            else:
                action = self.basic_strategy.get_action(hand.total, dealer_up, hand.is_soft)
            if action == 'S':
                i += 1
                continue
            elif action == 'H':
                card = self.shoe.deal_card()
                if card:
                    hand.add_card(card)
            elif action == 'D':
                if len(hand.cards) == 2:  # Double allowed on any two cards
                    hand.bet *= 2
                    card = self.shoe.deal_card()
                    if card:
                        hand.add_card(card)
                    i += 1
                    continue
                else:
                    # Invalid double: hit
                    card = self.shoe.deal_card()
                    if card:
                        hand.add_card(card)
            elif action == 'P':
                if len(self.player_hands) >= max_hands or not is_pair:
                    # Invalid split: hit
                    card = self.shoe.deal_card()
                    if card:
                        hand.add_card(card)
                    continue
                pair_rank = hand.cards[0].rank
                # Perform split
                second_card = hand.cards.pop()
                new_hand = Hand(bet=hand.bet, is_split=True)
                new_hand.add_card(second_card)
                card1 = self.shoe.deal_card()
                if card1:
                    hand.add_card(card1)
                card2 = self.shoe.deal_card()
                if card2:
                    new_hand.add_card(card2)
                self.player_hands.insert(i + 1, new_hand)
                if pair_rank == 11:
                    hand.is_split = True
                    i += 1
                    continue
                # Continue evaluating original hand
            else:
                i += 1
                continue

    def _play_dealer(self) -> None:
        """Dealer plays: hit until total >= 17 (stands on soft 17)."""
        while self.dealer_hand.total < 17:
            card = self.shoe.deal_card()
            if card:
                self.dealer_hand.add_card(card)

    def play_hand(self, bet: float) -> float:
        """
        Play complete hand with basic strategy. Returns total payout amount.
        Handles natural BJ early payout (3:2), doubles/splits as allowed.
        """
        if bet <= 0:
            return 0.0

        self.reset()
        self.shoe.reshuffle_if_needed(self.config['penetration'])

        # Deal initial cards: player 2, dealer up + hole
        player_hand = Hand(bet=bet)
        self.player_hands = [player_hand]
        p1 = self.shoe.deal_card()
        p2 = self.shoe.deal_card()
        if p1: player_hand.add_card(p1)
        if p2: player_hand.add_card(p2)
        d_up = self.shoe.deal_card()
        d_hole = self.shoe.deal_card()
        if d_up: self.dealer_hand.add_card(d_up)
        if d_hole: self.dealer_hand.add_card(d_hole)

        # Check natural BJ
        if len(player_hand.cards) == 2 and player_hand.is_blackjack():
            if self.dealer_hand.is_blackjack():
                return 0.0  # push
            return 1.5 * bet

        # Check dealer BJ (player not BJ)
        if self.dealer_hand.is_blackjack():
            return -bet

        # Player actions
        self._play_player_hands()

        # Dealer actions
        self._play_dealer()

        # Resolve all player hands (including splits/doubles)
        total_payout = sum(self._resolve_hand(ph) for ph in self.player_hands)
        return total_payout

        #