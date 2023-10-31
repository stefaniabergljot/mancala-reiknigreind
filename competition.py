from dataclasses import dataclass
import importlib.machinery
from itertools import permutations
from pathlib import Path
import types

import pandas as pd

from mancala.game import game as play_game, ActionFunction


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
matches = permutations(groups, 2)
results = []
for home, away in matches:
    result = play_game(home.action, away.action)
    results.append((home.name, away.name, result))

result_df = pd.DataFrame(results, columns=['home', 'away', 'result'])
print(result_df)