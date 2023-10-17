import array
import random
from typing import Tuple


NAME = "RANDOM"


def action(board: array.array, legal_actions: Tuple[int, ...], player: int) -> int:
    "Let's just pick a random action"
    return random.choice(legal_actions)
