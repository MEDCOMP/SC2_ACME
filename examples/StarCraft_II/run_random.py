"""Random Agent for StarCraft II"""


from absl import app
from absl import flags
from pysc2.env import sc2_env
from pysc2.lib import features
from acme.wrappers import pysc2_wrapper_reaver as pysc2_wrapper
import acme
import dm_env
from acme import specs
from acme import core
# from acme.agents.StarCraft2 import random_agent
from acme import sc2_types
import numpy as np
from acme.agents.StarCraft2 import random_actor



flags.DEFINE_string('map_name', 'MoveToBeacon', 'which mini-game to play.')
flags.DEFINE_integer('num_episodes', 2, 'Number of episodes to train for')
flags.DEFINE_integer('save_replay_episodes', 0, "Save a replay of each episode after this many episodes. Default of 0 means don't save replays ")
flags.DEFINE_string('replay_dir', None, 'Directory to save replays. '
                                        'Linux distros will save on ~/StarCraftII/Replays/ + path(replay_dir)'
                                        'Windows distros will save on path(replay_dir)')

FLAGS = flags.FLAGS

def make_environment(evaluation: bool = False) -> dm_env.Environment:

    env = sc2_env.SC2Env(
        map_name=FLAGS.map_name,
        players=[sc2_env.Agent(sc2_env.Race.terran)],
        agent_interface_format=[features.parse_agent_interface_format(
            feature_screen=16,
            feature_minimap=16,
            rgb_screen=None,
            rgb_minimap=None
        )],
        step_mul=8,
        save_replay_episodes=FLAGS.save_replay_episodes,
        replay_dir=FLAGS.replay_dir
    )

    env = pysc2_wrapper.Pysc2Wrapper(env)

    return env

  
def main(_):
    env = make_environment()
    env_spec = acme.make_environment_spec(env)
    agent = random_actor.SC2RandomActor(env_spec)

    loop = acme.EnvironmentLoop(env, agent)
    loop.run(FLAGS.num_episodes)

if __name__ == '__main__':
    app.run(main)