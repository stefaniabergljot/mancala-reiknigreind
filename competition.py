import argparse
from collections import defaultdict
from dataclasses import dataclass
import importlib.machinery
from itertools import permutations
from pathlib import Path
import types

import pandas as pd

from mancala.game import game as play_game, ActionFunction, NoActionException, IllegalActionException


parser = argparse.ArgumentParser()
parser.add_argument('-N', '--number-of-rounds', default=10, help="How many double-rounds to play.", type=int)
args = parser.parse_args()
N = args.number_of_rounds


@dataclass
class Group:
    name: str
    action: ActionFunction


# Get path to all groups
p = Path('mancala/groups')
group_dirs = [str(x) for x in p.iterdir() if x.is_dir()]


# Load each group module and assign its name and action to a Group object.
groups = []
for group in group_dirs:
    group_name = group.split('/')[-1]
    if group_name == 'example_group':
        continue
    loader = importlib.machinery.SourceFileLoader('group_name', group + '/action.py')
    mod = types.ModuleType(loader.name)
    try:
        loader.exec_module(mod)
    except Exception as e:
        print(f'Excluding group {group_name}. Could not load its module:\n{e}')
    else:
        try:
            name = mod.NAME
        except Exception:
            name = group_name
        groups.append(Group(name, mod.action))


# Let everybody play against everbody (two rounds, home and away).
matches = tuple(permutations(groups, 2))
index = pd.MultiIndex.from_tuples((p0.name, p1.name) for p0, p1 in matches)
columns=[0, 1, -1]
results = pd.DataFrame(0, index=index, columns=columns)


expelled = []
for i in range(N):
    for player0, player1 in matches:
        if player0 in expelled or player1 in expelled:
            continue
        try:
            result = play_game(player0.action, player1.action)
        except (NoActionException, IllegalActionException) as e:
            player = player0 if e.player == 0 else player1
            legal_actions = e.legal_actions
            if isinstance(e, NoActionException):
                print(f'Expelling {player.name} for causing exception by failing to choose action as player{e.player} when {legal_actions=}, causing the exception: {str(e)}')
            else:
                action = e.action
                print(f'Expelling {player.name} for causing exception after choosing {action=} as player{e.player} when {legal_actions=} and causing the exception: {str(e)}')
            expelled.append(player)
            continue

        results.loc[(player0.name, player1.name), result] += 1


results.columns = ['player0', 'player1', 'draw']
for player in expelled:
    print(f'\nPlayer {player.name} was expelled.\n')
print(results, '\n')

scores = defaultdict(float)
for index in results.index:
    p0, p1 = index
    draw_score = 0.5  * float(results.loc[index, 'draw'])
    scores[p0] += float(results.loc[index, 'player0']) + draw_score
    scores[p1] += float(results.loc[index, 'player1']) + draw_score




max_length = max(len(key) for key in scores)
print('Score'.rjust(max_length+8))
print((max_length + 8) * '-')

print('\n'.join(f'{name.ljust(max_length)}{str(score).rjust(8)}' for name, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)))
print((max_length + 8) * '-')
