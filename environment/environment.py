from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tf_agents.specs import array_spec
from tf_agents.environments import py_environment
import numpy

tf.compat.v1.enable_v2_behavior()


class ZebroEnvironment(py_environment.PyEnvironment):

    """
    Env:
        Matrix of
            1 bit -> blocked
            1 bit -> speed
            3 bits -> important
            3 bit -> sunny
            8 bits -> timestamp
            total 16 bits uint16

    Action:
        8 -> North, South, East, West, Northeast ...
        1 -> Wait/Sleep

    """
    def __init__(self):
        super().__init__()
        self.map = None
        self.base = {"x": 0, "y": 0}
        self.zebros = [
            {"x": self.base["x"], "y": self.base["y"], "battery": 1.0, "damage": 0.0}]

        self.turn = 0 # to indicate which agent will take next action
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(),
            dtype=numpy.int32,
            minimum=0,
            maximum=8,
            name="action"
        )
        number_of_observation_points = self.map.size + len(self.zebros) * len(self.zebros[0].keys()) + len(self.base.keys())
        # TODO: Add a size variable for map
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(number_of_observation_points, 1),
            dtype=numpy.int16,
            minimum=0,
            name="observation"
        )

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        pass

    def _step(self, action):
        if self.zebros[self.turn]["battery"] == 0 or self.zebros[self.turn]["damage"] >= 1.0:
            pass

        if action == 0:
            pass

        """NORTH"""
        if action == 1:
            self.zebros[self.turn]["y"] -= 1

        """NORTHEAST"""
        if action == 2:
            self.zebros[self.turn]["x"] += 1
            self.zebros[self.turn]["y"] -= 1

        """EAST"""
        if action == 3:
            self.zebros[self.turn]["x"] += 1

        """SOUTHEAST"""
        if action == 4:
            self.zebros[self.turn]["x"] += 1
            self.zebros[self.turn]["y"] += 1

        """SOUTH"""
        if action == 5:
            self.zebros[self.turn]["y"] += 1

        """SOUTHWEST"""
        if action == 6:
            self.zebros[self.turn]["x"] -= 1
            self.zebros[self.turn]["y"] += 1

        """WEST"""
        if action == 7:
            self.zebros[self.turn]["x"] -= 1

        """NORTHWEST"""
        if action == 8:
            self.zebros[self.turn]["x"] -= 1
            self.zebros[self.turn]["y"] -= 1

        self.zebros[self.turn]["battery"] -= random.uniform(0.01, 0.025)
        self.zebros[self.turn]["damage"] += random.uniform(0.005, 0.01)

        turn = turn + 1
        turn = turn % len(self.zebros)

        return

    def _init_map(self, m, n):
        """

        :param m: the width of the map
        :param n: the height of the map
        :return: An M by N matrix of uint6 as a Map for the environment
        """
        pass

    def _calc_reward(self, map):
        """

        :param map: A numpy matrix representing the map
        :return: Integer representing the reward for the given map
        """
        pass

    def _end_state(self, map):
        """
        Checks if whole maps is explored.
        :param map: Numpy array representing the map.
        :return: true if the map is totally explored.
        """

        pass

    def _render(self):
        """
        Creates an image of the given environment
        :return: an
        """


a = ZebroEnvironment()