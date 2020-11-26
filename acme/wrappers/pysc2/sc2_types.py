"""
Data types used for pysc2 environment
We don't use the NestedSpec, which is a data type of acme, because it is less expressive than customized classes below
"""

import numpy as np
from typing import List
from pysc2.lib import actions


class Space:
    """
    Holds information about any generic space
    In essence is a simplification of gym.spaces module into a single endpoint
    """
    def __init__(self, shape=(), dtype=np.int32, domain=(0, 1), categorical=False, name=None):
        self.name = name
        self.shape, self.dtype = shape, dtype
        self.categorical, (self.lo, self.hi) = categorical, domain

    def is_discrete(self) -> bool:
        """
        Space is considered continuous if its values are only ints
        """
        return np.issubdtype(self.dtype, np.integer)

    def is_continuous(self) -> bool:
        """
        Space is considered continuous if its values can be floats
        """
        return np.issubdtype(self.dtype, np.floating)

    def is_spatial(self) -> bool:
        """
        Space is considered spacial if it has three-dimensional shape HxWxC
        """
        return len(self.shape) > 1 or type(self.hi) in [list, tuple]

    def size(self) -> int:
        """
        Number of labels if categorical
        Number of intervals if discrete (can have multiple in one space)
        Number of mean and log std.dev if continuous

        Meant to be used to determine size of logit outputs in models
        """
        if self.is_discrete() and self.categorical:
            if self.is_spatial():
                return self.hi
            return self.hi - self.lo

        sz = 1
        if len(self.shape) == 1:
            sz = self.shape[0]

        return sz

    def sample(self, n=1):
        """
        Sample from this space. Useful for random agent, for example.
        """
        if self.is_discrete():
            return np.random.randint(self.lo, self.hi+1, (n, ) + self.shape)

        if self.is_continuous():
            return np.random.uniform(self.lo, self.hi+1e-10, (n, ) + self.shape)

    def __repr__(self):
        mid = str(self.shape)
        if self.categorical:
            mid += ", cat: " + str(self.hi)
        return "Space(%s, %s, %s)" % (self.name, mid, str(self.dtype).strip("<class>' "))


class SC2Space(Space):
    def __init__(self, shape, name, spatial_feats=None, spatial_dims=None):
        """
        Example of one space with spatial features:
        name: screen{player_relative, selected, visibility_map, unit_hit_points_ratio, unit_density}
        """
        if spatial_feats:
            name += "{%s}" % ", ".join(spatial_feats)
        self.spatial_feats, self.spatial_dims = spatial_feats, spatial_dims

        super().__init__(shape, name=name)


class SC2FuncIdSpace(Space):
    def __init__(self, func_ids, args):
        super().__init__(domain=(0, len(func_ids)), categorical=True, name="function_id")
        self.args_mask = []
        for fn_id in func_ids:
            fn_id_args = [
                arg_type.name for arg_type in actions.FUNCTIONS[fn_id].args]
            self.args_mask.append([arg in fn_id_args for arg in args])


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
