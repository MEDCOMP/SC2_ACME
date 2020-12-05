import dm_env
from acme import specs
from acme import core
from acme import sc2_spec
from acme import sc2_types
from acme import adders
from acme.adders import reverb as reverb_adders
import numpy as np
import reverb

from pysc2.lib import named_array

from typing import Optional

class SC2RandomActor(core.Actor):
  """Fake actor that generates random actions for StarCraft II environment"""
  
  def __init__(
      self, spec:specs.EnvironmentSpec,
      adder: Optional[adders.Adder] = None,
      priority_exponent: float = 0.6,
      max_replay_size: int = 1000000,
      n_step: int = 1,
      discount: float = 0.99,
  ):
    
    self._spec = spec
    self.num_updates = 0

    if adder != None:
      self._adder = adder 
    else:

      # Create a replay server to add data to.
      replay_table = reverb.Table(
          name=reverb_adders.DEFAULT_PRIORITY_TABLE,
          sampler=reverb.selectors.Prioritized(priority_exponent),
          remover=reverb.selectors.Fifo(),
          max_size=max_replay_size,
          rate_limiter=reverb.rate_limiters.MinSize(1),
          signature=reverb_adders.SC2NStepTransitionAdder.signature(spec)
          )
      self._server = reverb.Server([replay_table], port=None)

      # The adder is used to insert observations into replay.
      address = f'localhost:{self._server.port}'

       # Create a reverb adder for random actor
      adder = reverb_adders.SC2NStepTransitionAdder(
          client=reverb.Client(address),
          n_step=n_step,
          discount=discount)
      
      self._adder = adder

  def select_action(self, observation:sc2_spec.Spec):
      action_spec = self._spec.actions

      available_actions_list = observation[2]
      # the index of action in "action_ids" defined in the environment wrapper
      available_actions_index = np.argwhere(available_actions_list > 0).flatten()
      function_id = np.random.choice(available_actions_index)

      non_function_id_spaces = action_spec.spaces[1:]
      args = []
      for arg in non_function_id_spaces:
        # spatial arguments have two dimensions
        if len(arg.shape) == 1:
          random_arg_0 = np.random.randint(0, arg.maximum)
          random_arg_1 = np.random.randint(0, arg.maximum)
          random_arg = [random_arg_0] + [random_arg_1]
        else:
          random_arg = np.random.randint(0, arg.maximum)

        # use append instead of extend here, spatial parameters like [x,y] coordinates should be contained in the same list
        args.append(random_arg)

      return [function_id] + args

  def observe_first(self, timestep: dm_env.TimeStep):
    if self._adder:
      self._adder.add_first(timestep)
    

  def observe(self, action, next_timestep: dm_env.TimeStep):
    if self._adder:
      self._adder.add(action, next_timestep)

  def update(self):
    self.num_updates += 1


