"""
Human player that takes input from the command line.
Ideally the function running the game should print the boards
so it's clearer to the human player what action the opponent took
"""
import array
from typing import Tuple

from mancala.game import copy_board, flip_board, board_repr  # noqa: F401

NAME = "Human"

do_print = True

def action(board: array.array, legal_actions: Tuple[int, ...], player: int) -> int:
    if do_print:
        print(board_repr(board, -1))
    print("Choose an action as player " + str(player) + " (" + str(legal_actions) + "): ")
    while True:
        try:
            action = int(input())
            if action in legal_actions:
                return action
        except Exception:
            print("Invalid action, try again")

