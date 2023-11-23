from array import array
from typing import Tuple, List

from mancala.game import copy_board, play_turn
from mancala.groups.minmax.action import alphabeta, MAX_SCORE, MIN_SCORE


NAME = 'Deterministic MinMax'


def deterministic_action(board: array, legal_actions: Tuple[int, ...], player: int, depth: int = 9) -> int:
    alpha = MIN_SCORE
    beta = MAX_SCORE
    rewards: List[Tuple[int, int]] = []
    for action in legal_actions:
        child = copy_board(board)
        next_player = play_turn(child, player, action)
        score = alphabeta(child, depth, player, alpha, beta, next_player)
        rewards.append((action, score))

    chosen_action, max_score = max(rewards, key=lambda e: e[1])
    return chosen_action


def minmax_action_10(board: array, legal_actions: Tuple[int, ...], player: int) -> int:
    return deterministic_action(board, legal_actions, player, 10)
