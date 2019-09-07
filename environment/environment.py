from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tf_agents.specs import array_spec
from tf_agents.environments import py_environment
import numpy
from math import sqrt

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
    def __init__(self, map_shape):
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
        self.MAX_RANGE = 15/100 * (map_shape[0] + map_shape[1])/2;


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

    def _reward_helper(self):
        """
        Helper function for the reward calculator, keeps zebro connected t
        o the others and makes sure they spread as much as possible.
        :param map:
        :return: 0 - if zebro out of rangem, it should refocus on finding the others;
                1 - if all zebros are at maximum spread, meaning its ok
                0,5 - needs to spread out more

        """
        dist_sum = 0
        count = 0
        min = self.MAX_RANGE + 1
        for i in range(0,len(self.zebros)):
            dist_x = abs(self.zebros[i]["x"]-self["x"])
            dist_y = abs(self.zebros[i]["y"] - self["x"])
            dist_to_zebro = sqrt(dist_x**2 + dist_y**2)
            if dist_to_zebro <= self.MAX_RANGE:
                count += 1
                dist_sum += dist_to_zebro
                min = min if min < dist_sum else dist_sum
        if count == 0:
            return 0
        avg = dist_sum / count
        if avg == self.MAX_RANGE:
            return 1
        if min == 0:
            return 0,5
        #if avg < MAX_RANGE:
        #    return 0.8
        return 0.8

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