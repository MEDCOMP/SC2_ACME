"""
Environment Specificiation used for pysc2 environment
We don't use the "specs.py" provided by acme because they are less expressive.
"""

import numpy as np
from typing import List
from pysc2.lib import actions
from acme.sc2_types import Space

class Spec:
    """
    Convenience class to hold a list of spaces, can be used as an iterable
    A typical environment is expected to have one observation spec and one action spec

    Note: Every spec is expected to have a list of spaces, even if there is only one space
    """
    def __init__(self, spaces: List[Space], name=None):
        self.name, self.spaces = name, spaces
        for i, space in enumerate(self.spaces):
            if not space.name:
                space.name = str(i)

    def sample(self, n=1):
        return [space.sample(n) for space in self.spaces]

    def __repr__(self):
        return "Spec: %s\n%s" % (self.name, "\n".join(map(str, self.spaces)))

    def __iter__(self):
        return (space for space in self.spaces)

    def __len__(self):
        return len(self.spaces)