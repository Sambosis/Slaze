"""
Utility functions for BlackjackBetOptimizer project.
Provides basic strategy lookup tables and training statistics computation.
"""

import numpy as np
from typing import Dict, Tuple, List, Any


def get_basic_strategy_table() -> Tuple[Dict[int, Dict[int, str]], Dict[int, Dict[int, str]], Dict[int, Dict[int, str]]]:
    """
    Returns pre-defined perfect basic strategy lookup tables for 4-8 decks,
    dealer stands on soft 17 (S17), double on any two cards, splits up to 4 hands (aces once).

    Returns:
        Tuple of (table_hard, table_soft, table_split):
        - table_hard: dict[int (5-20), dict[int (2-11), str]] 'H'/'S'/'D'
        - table_soft: dict[int (13-20), dict[int (2-11), str]] 'H'/'S'/'D'
        - table_split: dict[int (2-11 pair value), dict[int (2-11), str]] 'P' (split) or 'H' (don't split, play basic)
    
    Tables are hardcoded from standard multi-deck S17 basic strategy charts.
    Dealer upcard: 2-9=face, 10=10/J/Q/K, 11=A.
    """
    # Hard totals table (5-20 vs dealer 2-A)
    table_hard: Dict[int, Dict[int, str]] = {}
    
    # 5-8: always Hit
    for total in range(5, 9):
        table_hard[total] = {up: 'H' for up in range(2, 12)}
    
    # 9: Double vs 3-6
    table_hard[9] = {2: 'H', 3: 'D', 4: 'D', 5: 'D', 6: 'D', 7: 'H', 8: 'H', 9: 'H', 10: 'H', 11: 'H'}
    
    # 10: Double vs 2-9
    table_hard[10] = {2: 'D', 3: 'D', 4: 'D', 5: 'D', 6: 'D', 7: 'D', 8: 'D', 9: 'D', 10: 'H', 11: 'H'}
    
    # 11: Double vs 2-10
    table_hard[11] = {2: 'D', 3: 'D', 4: 'D', 5: 'D', 6: 'D', 7: 'D', 8: 'D', 9: 'D', 10: 'D', 11: 'H'}
    
    # 12: Stand vs 4-6
    table_hard[12] = {2: 'H', 3: 'H', 4: 'S', 5: 'S', 6: 'S', 7: 'H', 8: 'H', 9: 'H', 10: 'H', 11: 'H'}
    
    # 13-16: Stand vs 2-6
    for total in range(13, 17):
        table_hard[total] = {2: 'S', 3: 'S', 4: 'S', 5: 'S', 6: 'S', 7: 'H', 8: 'H', 9: 'H', 10: 'H', 11: 'H'}
    
    # 17-20: always Stand
    for total in range(17, 21):
        table_hard[total] = {up: 'S' for up in range(2, 12)}
    
    # Soft totals table (13-20 vs dealer 2-A)
    table_soft: Dict[int, Dict[int, str]] = {}
    
    # Standard soft actions lists indexed by dealer up 2-11
    soft_rows = [
        # 13 (A2): H H H D D H H H H H
        ['H', 'H', 'H', 'D', 'D', 'H', 'H', 'H', 'H', 'H'],
        # 14 (A3): H H D D D H H H H H
        ['H', 'H', 'D', 'D', 'D', 'H', 'H', 'H', 'H', 'H'],
        # 15 (A4): H D D D D H H H H H
        ['H', 'D', 'D', 'D', 'D', 'H', 'H', 'H', 'H', 'H'],
        # 16 (A5): H D D D D H H H H H
        ['H', 'D', 'D', 'D', 'D', 'H', 'H', 'H', 'H', 'H'],
        # 17 (A6): D D D D D H H H H H
        ['D', 'D', 'D', 'D', 'D', 'H', 'H', 'H', 'H', 'H'],
        # 18 (A7): S D D D D S S H H H
        ['S', 'D', 'D', 'D', 'D', 'S', 'S', 'H', 'H', 'H'],
        # 19 (A8): S D D D D S S S H H
        ['S', 'D', 'D', 'D', 'D', 'S', 'S', 'S', 'H', 'H'],
        # 20 (A9): S S S S S S S S S S
        ['S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S']
    ]
    
    dealer_ups = list(range(2, 12))
    for idx, total in enumerate(range(13, 21)):
        table_soft[total] = dict(zip(dealer_ups, soft_rows[idx]))
    
    # Split table (pair value 2-11 vs dealer 2-A): 'P' split, 'H' don't split (fallback to hard/soft)
    table_split: Dict[int, Dict[int, str]] = {}
    
    # 2s and 3s: P vs 2-7
    row_23 = {2: 'P', 3: 'P', 4: 'P', 5: 'P', 6: 'P', 7: 'P', 8: 'H', 9: 'H', 10: 'H', 11: 'H'}
    table_split[2] = row_23.copy()
    table_split[3] = row_23.copy()
    
    # 4s: P vs 5-6
    table_split[4] = {2: 'H', 3: 'H', 4: 'H', 5: 'P', 6: 'P', 7: 'H', 8: 'H', 9: 'H', 10: 'H', 11: 'H'}
    
    # 5s: never split (double as 10)
    table_split[5] = {up: 'H' for up in range(2, 12)}
    
    # 6s: P vs 2-6
    table_split[6] = {2: 'P', 3: 'P', 4: 'P', 5: 'P', 6: 'P', 7: 'H', 8: 'H', 9: 'H', 10: 'H', 11: 'H'}
    
    # 7s: P vs 2-7
    table_split[7] = {2: 'P', 3: 'P', 4: 'P', 5: 'P', 6: 'P', 7: 'P', 8: 'H', 9: 'H', 10: 'H', 11: 'H'}
    
    # 8s: always split
    table_split[8] = {up: 'P' for up in range(2, 12)}
    
    # 9s: P vs 2-6,8,9; H vs 7,10,A
    table_split[9] = {2: 'P', 3: 'P', 4: 'P', 5: 'P', 6: 'P', 7: 'H', 8: 'P', 9: 'P', 10: 'H', 11: 'H'}
    
    # 10s: never split
    table_split[10] = {up: 'H' for up in range(2, 12)}
    
    # Aces (11): always split (once)
    table_split[11] = {up: 'P' for up in range(2, 12)}
    
    return table_hard, table_soft, table_split


def compute_stats(
    bankroll_history: List[float],
    hands: int,
    wins: int,
    losses: int,
    bets: List[float]
) -> Dict[str, float]:
    """
    Computes key training statistics for display and logging.

    Args:
        bankroll_history: List of bankroll values after each hand.
        hands: Total number of hands played.
        wins: Number of winning hands (excludes pushes).
        losses: Number of losing hands (excludes pushes).
        bets: List of all bet amounts placed.

    Returns:
        Dict with:
        - 'avg_return': Average return per hand (total profit / hands).
        - 'win_ratio': Win rate (wins / (wins + losses)).
        - 'avg_bet': Average of last 1000 bets (or all if fewer).
        - 'max_drawdown': Maximum peak-to-trough bankroll drop.

    Handles empty lists gracefully with 0.0 defaults.
    """
    stats: Dict[str, float] = {}
    
    # Average return per hand
    if bankroll_history and len(bankroll_history) > 1 and hands > 0:
        total_profit = bankroll_history[-1] - bankroll_history[0]
        stats['avg_return'] = total_profit / hands
    else:
        stats['avg_return'] = 0.0
    
    # Win ratio (ignores pushes)
    total_decisive = wins + losses
    stats['win_ratio'] = wins / total_decisive if total_decisive > 0 else 0.0
    
    # Average bet size (recent 1000)
    recent_bets = bets[-1000:]
    stats['avg_bet'] = np.mean(recent_bets) if recent_bets else 0.0
    
    # Maximum drawdown (peak-to-trough)
    stats['max_drawdown'] = 0.0
    if len(bankroll_history) > 1:
        peak = bankroll_history[0]
        drawdown = 0.0
        for bankroll in bankroll_history:
            if bankroll > peak:
                peak = bankroll
            drawdown = max(drawdown, peak - bankroll)
        stats['max_drawdown'] = drawdown
    
    return stats