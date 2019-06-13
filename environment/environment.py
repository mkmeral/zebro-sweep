from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tf_agents.environments import py_environment

tf.compat.v1.enable_v2_behavior()


class ZebroEnvironment(py_environment.PyEnvironment):

    def __init__(self):
        super().__init__()
        self._action_spec = None  # TODO: add action spec
        self._observation_spec = None  # TODO: add observation spec

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        pass

    def _step(self, action):
        pass
