""" Wraps a pysc2 environment to be used as a dm_env envrionment."""

from typing import List

from acme import specs
from acme import types

import dm_env
import pysc2 
import numpy as np
from pysc2.env import sc2_env
from pysc2.lib import features
from pysc2.lib import actions

class Pysc2Wrapper(dm_env.Environment):
    """Environment warpper for DeepMind pysc2 mini-game envrionment."""

    # Note: we don't inherit from base.EnvironmentWrapper because that class
    # assumes that the wrapped environment is a dm_env.Environment.

    def __init__(self, environment: sc2_env.SC2Env):
        self._envrionment = environment
        self._reset_next_step = True

        # Convert the first agent's action and observation specs.
        observation_spec_list = environment.observation_spec()
        action_spec_list = environment.action_spec()
        obs_spec = observation_spec_list[0]
        action_spec = action_spec_list[0]
        self._observation_spec = _convert_obs_to_spec(obs_spec)
        self._action_spec = _convert_act_to_spec(action_spec) 

    def reset(self) -> dm_env.TimeStep:
        """Resets the episode """
        self._reset_next_step = False
        observation = self._envrionment.reset()
        return dm_env.restart(observation)

    def step(self, action: types.NestedSpec) -> dm_env.TimeStep:
        """
        The first element of action is the function id, the others are corresponding arguments.
        For example:
        select_rect(select_add, (x1, y1), (x2, y2)) --> [3, [[0], [3,1], [11,2]]]
        """
        fn_id = action[0]
        args = action[1]

        action = [actions.FunctionCall(fn_id, args)]
        timestep = self._envrionment.step(action)
        return timestep
                
    def observation_spec(self) -> types.NestedSpec:
        return self._observation_spec

    def action_spec(self) -> types.NestedSpec:
        return self._action_spec
    
    def close(self):
        self._envrionment.close()


def _convert_obs_to_spec(NamedDict: pysc2.lib.named_array.NamedDict) -> types.NestedSpec:
    """Converts a pysc2 NamedDict to a dm_env spec or nested structure of specs.

    Get the NamedDict from pysc2 obs_spec and convert the value into spec.Array,
    the key is the feature name.

    Args:
        NamedDict: The pysc2 NamedDict to convert.
    
    Returns:
        A dm_env spec or nested structure of specs, corresponding to the input NamedDict.
    """
    
    # extract data type information for each field in observation spec from features.py in pysc2.lib
    dtype_dict = {
        "map_name": str,
        "home_race_requested": np.int32,
        "away_race_requested": np.int32,
        "feature_screen": np.int32,
        "feature_minimap": np.int32,
        "rgb_screen": np.int32,
        "rgb_minimap": np.int32,
        "last_actions": np.int32吉泽明步 在线,
        "action_result": np.int32,
        "alerts": np.int32,
        "game_loop": np.int32,
        "score_cumulative": np.int32,
        "score_by_category": np.int32,
        "score_by_vital": np.int32,
        "player": np.int32,
        "control_groups": np.int32,
        "single_select": np.int32,
        "multi_select": np.int32,
        "cargo": np.int32,
        "cargo_slots_available": np.int32,
        "build_queue": np.int32,
        "production_queue": np.int32,
        "feature_units": np.int64,
        "feature_effects": np.int32,
        "raw_units": np.int64,
        "raw_effects": np.int32,
        "upgrades": np.int32,
        "unit_counts": np.int32,
        "camera_position": np.int32,
        "camera_size": np.int32,
        "available_actions": np.int32,
        "radar": np.int32,
    }

    obs_spec = {}
    for key, value in NamedDict.items():
        spec_array = specs.Array(
            shape=value,
            dtype=dtype_dict[key],
            name=key
        )
        obs_spec[key] = spec_array

    return obs_spec

def _convert_act_to_spec(valid_actions: pysc2.lib.actions.ValidActions) -> types.NestedSpec:
    action_spec = {}

    for fld in valid_actions.types._fields:
        arg = getattr(valid_actions.types,fld)
        spec_array = specs.BoundedArray(
            shape=arg.sizes,
            dtype=np.int32,
            minimum=0,
            maximum=arg.sizes[0],
            name=arg.name)
        action_spec[arg.name] = spec_array
    
    num_of_fn_ids = len(valid_actions.functions)
    spec_array = specs.Array(
        shape=(num_of_fn_ids, ),
        dtype=np.int32,
        name="function_id"
    )

    action_spec["function_id"] = spec_array

    return action_spec


class ObservationWrapper:
    def __init__(self, _features=None, action_ids=None):
        self.spec = None


