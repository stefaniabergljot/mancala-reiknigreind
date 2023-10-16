"""Example action.py.

Create a directory inside mancala/groups
and place all your code inside that directory.
Create a file called action.py within that directory.
We will only use the function action defined in mancala/group/your_group/action.py.
"""
import array
from typing import Tuple


NAME = "The name of your group "


def action(board: array.array, legal_actions: Tuple[int, ...]) -> int:
    # Your action, do somthing smart.
    # Make sure not to change the board.
    # If you need to change it, copy it:
    # from copy import deepcopy
    # board = deepcopy(board)
    raise NotImplementedError
