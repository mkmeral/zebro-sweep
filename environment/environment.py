from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import tensorflow as tf
from tf_agents.specs import array_spec
from tf_agents.environments import py_environment
from tf_agents.trajectories import time_step as ts

import numpy
from math import sqrt
import random
from .map import Map
tf.compat.v1.enable_v2_behavior()


# noinspection PyCompatibility
class ZebroEnvironment(py_environment.PyEnvironment):

    DEBUG_ENV = True

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
    def __init__(self, map_shape, number_of_zebros=5, step_size=1, visible_radius=1, render=False):
        super().__init__()
        self.initialize(map_shape, number_of_zebros)
        self.step_size = 10
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(),
            dtype=numpy.int32,
            minimum=0,
            maximum=8,
            name="action"
        )
        self.MAX_RANGE = 15/100 * (map_shape[0] + map_shape[1])/2
        self.step_size = step_size
        self.diagonal_step_size = int(sqrt(step_size**2 / 2))
        self.visible_radius = visible_radius
        self.episode_ended = False
        self.render = render
        self.image_list = []

    def initialize(self, map_shape, number_of_zebros):
        self.map = Map(map_shape[0], map_shape[1])
        self.timestamp=0
        self.turn = 0   # to indicate which agent will take next action
        self.base = {"x": random.randint(0, map_shape[0]-1), "y": random.randint(0, map_shape[1]-1)}
        while (self.map.map[self.base["x"]][self.base["y"]] & 0b1000000000000000) > 0:
            self.base = {"x": random.randint(0, map_shape[0]-1), "y": random.randint(0, map_shape[1]-1)}

        self.zebros = []
        for _ in range(number_of_zebros):
            self.zebros.append({"x": self.base["x"], "y": self.base["y"], "battery": 1.0, "damage": 0.0})

        self.number_of_observation_points = map_shape[0] * map_shape[1] + len(self.zebros) * len(self.zebros[0].keys()) + len(self.base.keys())
        # TODO: Add a size variable for map
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(self.number_of_observation_points, 1),
            dtype=numpy.int32,
            minimum=0,
            name="observation"
        )

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        ZebroEnvironment.terminalOutput("RESET the environment")
        self.initialize(self.map.map.shape, len(self.zebros))
        return ts.restart(self._get_observation_vector())

    def _step(self, action):
        ZebroEnvironment.terminalOutput('---------------')
        ZebroEnvironment.terminalOutput("ACTION:", action)
        ZebroEnvironment.terminalOutput("TURN:", self.turn)
        ZebroEnvironment.terminalOutput("TIMESTAMP:", self.timestamp)
        ZebroEnvironment.terminalOutput("Zebro:", self.zebros[self.turn])

        if self.turn == 0:
            self.timestamp += 1

        if self._end_state():
            ZebroEnvironment.terminalOutput('END STATE has been reached/')
            return self._reset()

        if self.zebros[self.turn]["battery"] <= 0.0 \
                or self.zebros[self.turn]["damage"] >= 1.0\
                or action == 0:
            self.turn = (self.turn + 1) % len(self.zebros)
            return ts.transition(
                self._get_observation_vector(),
                0
            )

        zebro_curr_x = self.zebros[self.turn]["x"]
        zebro_curr_y = self.zebros[self.turn]["y"]

        """NORTH"""
        if action == 1:
            for i in range(self.step_size):
                if not self.map.square_is_blocked(zebro_curr_x, zebro_curr_y - 1):
                    zebro_curr_y -= 1
                else:
                    break

        """NORTHEAST"""
        if action == 2:
            for i in range(self.diagonal_step_size):
                if not self.map.square_is_blocked(zebro_curr_x + 1, zebro_curr_y - 1):
                    zebro_curr_x += 1
                    zebro_curr_y -= 1
                else:
                    break

        """EAST"""
        if action == 3:
            for i in range(self.step_size):
                if not self.map.square_is_blocked(zebro_curr_x + 1, zebro_curr_y):
                    zebro_curr_x += 1
                else:
                    break

        """SOUTHEAST"""
        if action == 4:
            for i in range(self.diagonal_step_size):
                if not self.map.square_is_blocked(zebro_curr_x + 1, zebro_curr_y + 1):
                    zebro_curr_x += 1
                    zebro_curr_y += 1
                else:
                    break

        """SOUTH"""
        if action == 5:
            for i in range(self.step_size):
                if not self.map.square_is_blocked(zebro_curr_x, zebro_curr_y + 1):
                    zebro_curr_y += 1
                else:
                    break

        """SOUTHWEST"""
        if action == 6:
            for i in range(self.diagonal_step_size):
                if not self.map.square_is_blocked(zebro_curr_x - 1, zebro_curr_y + 1):
                    zebro_curr_x -= 1
                    zebro_curr_y += 1
                else:
                    break

        """WEST"""
        if action == 7:
            for i in range(self.step_size):
                if not self.map.square_is_blocked(zebro_curr_x - 1, zebro_curr_y):
                    zebro_curr_x -= 1
                else:
                    break

        """NORTHWEST"""
        if action == 8:
            for i in range(self.diagonal_step_size):
                if not self.map.square_is_blocked(zebro_curr_x - 1, zebro_curr_y - 1):
                    zebro_curr_x -= 1
                    zebro_curr_y -= 1
                else:
                    break
        if self.zebros[self.turn]["x"] == zebro_curr_x and self.zebros[self.turn]["y"] == zebro_curr_y:
            ZebroEnvironment.terminalOutput("DID NOT MOVE")

        self.zebros[self.turn]["x"] = zebro_curr_x
        self.zebros[self.turn]["y"] = zebro_curr_y

        self.zebros[self.turn]["battery"] -= random.uniform(0.01, 0.025)
        self.zebros[self.turn]["damage"] += random.uniform(0.005, 0.01)

        # Recharge
        sun_value = (self.map.map[zebro_curr_x][zebro_curr_y] >> 12) & 0b111
        self.zebros[self.turn]["battery"] += sun_value * 0.005

        self.turn = (self.turn + 1) % len(self.zebros)

        reward = self.map.visit(
            (self.zebros[self.turn]["x"], self.zebros[self.turn]["y"]),
            self.timestamp,
            radius=self.visible_radius
        )

        reward *= self._reward_helper()

        if self.render:
            self.image_list.append(self.map._render())

        ZebroEnvironment.terminalOutput("REWARD:", reward)

        return ts.transition(self._get_observation_vector(), reward)

    def _reward_helper(self):
        """
        Helper function for the reward calculator, keeps zebro connected t
        o the others and makes sure they spread as much as possible.
        :return: 0 - if zebro out of rangem, it should refocus on finding the others;
                1 - if all zebros are at maximum spread, meaning its ok
                0,5 - needs to spread out more

        """
        dist_sum = 0
        count = 0
        min_reward = self.MAX_RANGE + 1
        for i in range(0, len(self.zebros)):
            dist_x = abs(self.zebros[i]["x"] - self.zebros[self.turn]["x"])
            dist_y = abs(self.zebros[i]["y"] - self.zebros[self.turn]["y"])
            dist_to_zebro = sqrt(dist_x**2 + dist_y**2)
            if dist_to_zebro <= self.MAX_RANGE:
                count += 1
                dist_sum += dist_to_zebro
                min_reward = min_reward if min_reward < dist_sum else dist_sum
        if count == 0:
            return 0
        avg = dist_sum / count
        if avg == self.MAX_RANGE:
            return 1
        if min_reward == 0:
            return 0.5
        #if avg < MAX_RANGE:
        #    return 0.8
        return 0.8

    def _end_state(self):
        """
        Checks if whole maps is explored.
        :return: true if the map is totally explored.
        """
        # Check if timestamp started to overflow
        if self.timestamp >= 0b1000000:
            return True

        # Check if there's still a functioning zebro
        still_alive = False
        for zebro in self.zebros:
            if zebro["battery"] > 0 and zebro["damage"] < 1.0:
                still_alive = True

        if not still_alive:
            return True

        # Check if all the pixels are discovered
        for i in range(0, self.map.map.shape[0]):
            for j in range(0, self.map.map.shape[1]):
                if self.map.map[i][j] & (2**6-1) == 0:
                    return False

        return True

    def _get_observation_vector(self):
        """
        Returns observation as a vector
        :return: Observation vector
        """

        observation = numpy.zeros((self.number_of_observation_points, 1), dtype=numpy.int32)

        np_array = numpy.asarray(self.map.map).reshape(-1)

        for i in range(len(np_array)):
            observation[i] = np_array[i]

        x = np_array.shape[0]

        # Add the base
        observation[x] = self.base["x"]
        observation[x+1] = self.base["x"]
        x += 2

        for z in range(len(self.zebros)):
            zebro_no = (z + self.turn) % len(self.zebros)
            observation[x] = self.zebros[zebro_no]["x"]
            observation[x] = self.zebros[zebro_no]["y"]
            observation[x] = int(self.zebros[zebro_no]["battery"]*100)
            observation[x] = int(self.zebros[zebro_no]["damage"]*100)
            x += 4

        return observation

    def _render(self):
        """
        Creates an image of the given environment
        :return: an
        """
        # Todo

    def terminalOutput(*args):
        if ZebroEnvironment.DEBUG_ENV:
            print(time.time(), args)
