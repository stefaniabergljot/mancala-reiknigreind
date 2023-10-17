""" An example of how you can play the game.

To play the game:
  1. Implement your agent's logic in mancala/groups/your_group/.
     In particular, provide an action function.
     NOTE: Your final action functions should be named action
     and be in in mancala/groups/your_group/action.py
     so that it can be import with
     from mancala.groups.your_group.action import action.
  2. Import the game from mancala.game and any action functions you wish to use.
  3. Play the game.
"""
from mancala.game import game
from mancala.groups.group_random.action import action as random_action


def play_one_game():
    result = game(random_action, random_action)
    if result == 0:
        print("Player 0 won")
    elif result == 1:
        print("Player 1 won")
    else:
        print("Draw")


if __name__ == "__main__":
    play_one_game()
