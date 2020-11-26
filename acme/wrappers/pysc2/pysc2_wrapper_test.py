"""Test for pysc2_wrapper"""

import unittest
from absl.testing import absltest
from absl.testing import parameterized
from acme.wrappers import pysc2_wrapper
from dm_env import specs
import numpy as np
from pysc2.env import sc2_env
from pysc2.lib import features
from acme import types
from acme import specs
from typing import Any, Callable, Iterable, Mapping, Union


class Pysc2WrapperTest(parameterized.TestCase):

    def test_MoveToBeacon(self):
        env = sc2_env.SC2Env(
            map_name='MoveToBeacon',
            players=[sc2_env.Agent(sc2_env.Race.terran)],
            agent_interface_format=[features.parse_agent_interface_format(
                feature_screen=16,
                feature_minimap=16,
                rgb_screen=None,
                rgb_minimap=None
            )],
        )

        env = pysc2_wrapper.Pysc2Wrapper(env)

        # observation_spec = env.observation_spec()
        # print(observation_spec)

        # action_spec = env.action_spec()
        # print(action_spec)

        # Test step
        timestep = env.reset()
        self.assertTrue(timestep.first())
        timestep = env.step([3, [[0], [3,1], [11,2]]])
        #print("True")
        env.close()
        
        


if __name__ == '__main__':
    absltest.main()


