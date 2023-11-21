from typing import Tuple, List
import numpy as np


#from mancala import *
from array import array

#weights = np.load('trained_weights.npy')
weights=[-0.41710062, -0.42309405,  0.08948978, -0.35645619, -0.00753666,  0.32431281]

def UpdateBoard(B, move):
    seeds = B[move]
    B[move] = 0
    i = move
    while seeds > 0:
        i = (i + 1) % 14
        if (move < 7 and i == 13) or (move > 6 and i == 6):  # Skip opponent's Mancala
            continue
        B[i] += 1
        seeds -= 1

def getLegalMoves(B, p):
    # Returns a list of legal moves for player p
    # A move is represented by the index of the pit the player chooses to start from
    if p == 1:
        return [i for i in range(6) if B[i] > 0]
    else:
        return [i for i in range(7, 13) if B[i] > 0]



def evaluate_heuristics(B, player):
# H1: Hoard as many seeds as possible in one pit.
  h1 = max(B[7:13]) if player == 1 else max(B[0:6])
# H2: Keep as many seeds on the player's own side.
  h2 = sum(B[7:13]) if player == 1 else sum(B[0:6])
# H3: Have as many moves as possible from which to choose.
  h3 = len(getLegalMoves(B, player))
# H4: Maximise the amount of seeds in a player's own store.
  h4 = B[13] if player == 1 else B[6]
# H5: Move the seeds from the pit closest to the opponent's side.
# For Player 1, this is index 12, and for Player 0, it is index 5.
  h5 = B[12] if player == 1 else B[5]
# H6: Keep the opponent's score to a minimum.
# This heuristic requires looking ahead two moves, which is complex.
# For simplicity, we'll use the current score of the opponent.
  h6 = -B[6] if player == 1 else -B[13]
  return [h1, h2, h3, h4, h5, h6]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def copy_board(board):
    return array("i", board)


def value_function(B, player, weights):
    features = evaluate_heuristics(B, player)
    weighted_sum = np.dot(weights, features)
    return sigmoid(weighted_sum)


def action(
    board: List[int],              # 14-element int array
    legal_actions: Tuple[int, ...],# tuple of board indexes
    player: int                    # 0 for player0, 1 for player 1
) -> int:
    best_action = None
    best_value = -float('inf')

    # Adjust player index to match your model's expectation (if necessary)
    #player = player + 1

    for action in legal_actions:
        #Simulate the move
        temp_board = copy_board(board)
        UpdateBoard(temp_board, action)
        
        # Evaluate the move
        move_value = value_function(temp_board, player, weights)

        # Choose the move with the highest value
        if move_value > best_value:
            best_value = move_value
            best_action = action

    return best_action if best_action is not None else -1

