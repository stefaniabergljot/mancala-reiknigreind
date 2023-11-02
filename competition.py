import argparse
from dataclasses import dataclass
import importlib.machinery
from itertools import permutations
from pathlib import Path
import types

import pandas as pd

from mancala.game import game as play_game, ActionFunction


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
    loader.exec_module(mod)
    groups.append(Group(group_name, mod.action))


# Let everybody play against everbody (two rounds, home and away).
matches = tuple(permutations(groups, 2))
index = pd.MultiIndex.from_tuples((p0.name, p1.name) for p0, p1 in matches)
columns=[0, 1, -1]
results = pd.DataFrame(0, index=index, columns=columns)

for i in range(N):
    for player0, player1 in matches:
        result = play_game(player0.action, player1.action)
        results.loc[(player0.name, player1.name), result] += 1


results.columns = ['player0', 'player1', 'draw']
print(results)