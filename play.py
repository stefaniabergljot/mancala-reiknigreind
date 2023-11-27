import random

from mancala.game import game as play_game
from competition import load_groups, Group
from mancala.groups.human.humanaction import action, board_repr


human = Group(name='human', action=action)
groups = load_groups()


print('Which player do you want to play against?')
for i, group in enumerate(groups):
    print(f'\t{i}: {group.name}')
print('Select number')
group_idx = int(input())

oppenent = groups[group_idx]
_opponent_action = oppenent.action


def oppenent_action(board, legal_actions, player):
    print(board_repr(board, -1))
    a = _opponent_action(board, legal_actions, player)
    print(f'{oppenent.name} chose {a}')
    return a


oppenent.action = oppenent_action


players = [human, oppenent]
random.shuffle(players)

result = play_game(players[0].action, players[1].action)

if result >= 0:
    print(f'{players[result].name} won')
else:
    print("its a draw")



