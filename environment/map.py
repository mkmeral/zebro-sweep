import math
import numpy as np
import random
from PIL import Image


class Map:
    """A 2D map to be used for Zebro Area sweeping simulations.

    The matrix consists of 16bit unsigned integers, which represents wide range of values as shown below.

    bsssuuiiiitttttt
    b : represents if the pixel is blocked. If so, all other values should be zero/unsignificant.
    s : represents slowness of the area. Can be considered as 1/speed. It will affect the speed of the vehicles moving on it.
    u : represents the sun luminosity. It will be determinant factor of the chargings speed of solar powered vehicles.
    i : represents the importance. Importance will affect the rewards the agent gains by visiting the pixel.
    tttttt : represents the last time step any vehicle has visited the area. It will affect the reward of the agent.

    """

    def __init__(self, width, height, blocked_ratio=0.01, slow_ratio=0.05, sun_avg=0.4, important_areas=None):
        """Generates a map object with the given parameters.

        Parameters
        ----------
        width : int
            The desired width of the map.
        height : int
            The desired height of the map.
        blocked_ratio : float, optional
            The desired ratio of blocked areas of the whole map.
        slow_ratio : float, optional
            The desired ratio of slowing areas (e.g. water) areas of the whole map.
        sun_avg : float, optional
            The desired average of sun values of all pixels.
        important_areas : list, optional
            The locations desired to be focused. So this is a list of tuples, containing coordinates.

        Returns
        -------
        list
            a list of strings representing the header columns.
        """

        # Generate base map
        self.map = np.zeros((height, width), dtype=np.int16)
        self.map += self.generate_blocks(blocked_ratio)
        self.map += self.generate_slow_areas(slow_ratio)
        self.map += self.generate_sunny_areas(sun_avg)
        if important_areas is not None:
            self.map += self.generate_important_areas(important_areas)

    def square_is_blocked(self, x, y):

        offset_for_blocked_value = 15
        b = self.map[x, y] & (1 << offset_for_blocked_value)

        # if not blocked
        if b == 0:
            return true
        return false

    def generate_blocks(self, blocked_ratio):
        """Generates a map (as matrix) containing only blocked areas.

        Parameters
        ----------
        blocked_ratio : float
            The desired ratio of blocked areas over whole map.

        Returns
        -------
        map
            a map as a matrix containing only the blocked areas.
        """

        height, width = self.map.shape

        while True:
            #blocks = np.zeros((height, width), dtype=np.int16)
            blocks = np.ones((height, width), dtype=np.int16) * 2
            offset_for_blocked_value = 15

            nr_blocked_squares = int(width * height * blocked_ratio)

            blocked_positions = random.sample(range(0, width * height - 1), nr_blocked_squares)

            # block random squares on the map
            for i in blocked_positions:
                blocks[int(i / width), int(i % width)] = 1

            if self.check_fully_blocked_areas(blocks):
                # Also shift bits into proper place
                for row in slow_areas:
                    for val in row:
                        blocks[row][val] = val << offset_for_blocked_value

                return blocks
            else:
                # should call this method again from outside
                return 0

    def generate_slow_areas(self, slow_ratio):
        """Generates a map (as matrix) containing only slowing areas which can be considered (in zebro example) as water.

        Parameters
        ----------
        slow_ratio : float
            The desired ratio of slowing areas over whole map.

        Returns
        -------
        map
            a map as a matrix containing only the slowing areas.
        """

        height, width = self.map.shape
        slow_areas = np.zeros((height, width), dtype=np.int16)

        offset_for_slow_value = 13
        max_value = 2 ** 3

        number_of_centers = 2  # TODO: Create a size based value for it

        # Create random sunny circular points
        for i in range(number_of_centers):
            x = random.randint(0, width)
            y = random.randint(0, height)

            slow_areas = + self.create_circular_effect((x, y), max_value=7)

        # Check to make sure we dont have an overflow!
        # Also shift bits into proper place
        for row in slow_areas:
            for val in row:
                if val > max_value:
                    val = max_value
                slow_areas[row][val] = val << offset_for_slow_value

        return slow_areas

    def generate_sunny_areas(self, sun_avg):
        """Generates a map (as matrix) containing only sunny areas.

        Parameters
        ----------
        sun_avg : float
            The desired average of sun values of all pixels.

        Returns
        -------
        map
            a map as a matrix containing only the sunny areas.
        """

        height, width = self.map.shape
        sunny_areas = np.zeros((height, width), dtype=np.int16)

        offset_for_sun_value = 10
        max_value = 2 ** 3

        number_of_centers = 8  # TODO: Create a size based value for it

        # Create random sunny circular points
        for i in range(number_of_centers):
            x = random.randint(0, width)
            y = random.randint(0, height)

            sunny_areas = + self.create_circular_effect((x, y), max_value=7)

        # Check to make sure we dont have an overflow!
        # Also shift bits into proper place
        for row in sunny_areas:
            for val in row:
                if val > max_value:
                    val = max_value
                sunny_areas[row][val] = val << offset_for_sun_value

        return sunny_areas

    def generate_important_areas(self, important_areas):
        """Generates a map (as matrix) of importance, containing only importance values.

        Parameters
        ----------
        important_areas : list
            The locations desired to be focused. So this is a list of tuples, containing coordinates.

        Returns
        -------
        map
            a map as a matrix containing only importance values.
        """

        height, width = self.map.shape
        importance_map = np.zeros((height, width), dtype=np.int16)

        max_importance_value = math.pow(2, 4) - 1

        for coordinate in important_areas:
            importance_map += self.create_circular_effect(coordinate, max_value=15)

        matrix_avg = importance_map.sum() / importance_map.size

        coefficient = matrix_avg / 2 if matrix_avg > (max_importance_value * 2 / 3) else 1

        for row in importance_map:
            for val in row:
                val *= coefficient
                importance_map[row][val] = max_importance_value if val > max_importance_value else val

        #   by making those locations the most important points
        #   and then slowly reducing the importance in radius r
        #   r should be chosen by the size of map and #important_locations
        #   to keep the average importance in mid values
        return importance_map

    def check_fully_blocked_areas(self, blocks):
        """Checks if any unblocked point A is reachable by all the other unblocked points. This is done to make sure episodes will end.

        Parameters
        ----------
        blocks : float
            A map that contains blocks.

        Returns
        -------
        boolean
            True if there are no unreachable areas.
        """

        height, width = self.map.shape

        # get first non blocked square
        i = 0
        for square in blocks.flat:
            if square == 2:
                break
            else:
                i += 1

        self.fill(blocks, width, height, int(i / width), int(i % width))

        for square in blocks.flat:
            if square == 2:
                return False

        return True
        # TODO: check if any point A is accessible by any other point B

    def fill(self, blocks, width, height, x, y):
        if blocks[x, y] == 2:
            blocks[x, y] = 0
            # recursively invoke fill on surrounding cells:
            if x > 0:
                self.fill(blocks, width, height, x - 1, y)
            if x < height - 1:
                self.fill(blocks, width, height, x + 1, y)
            if y > 0:
                self.fill(blocks, width, height, x, y - 1)
            if y < width - 1:
                self.fill(blocks, width, height, x, y + 1)

    def create_circular_effect(self, coordinates, radius=None, max_value=1.0):
        """Generates a circular patterned map with the given parameters.

        Parameters
        ----------
        coordinates : tuple
            coordinates for the center of the circle
        radius : int, optional
            the max distance to add value from the center.
        max_value : float, optional
            the max value to add.

        Returns
        -------
        np.matrix
            A map containing the newly added values in a circular pattern.
        """

        MAGICAL_PROPORTION_FOR_RADIUS = 6

        height, width = self.map.shape
        result = np.zeros((height, width))

        if radius is None:
            radius = (height + width) / (2 * MAGICAL_PROPORTION_FOR_RADIUS)

        min_x = 0 if radius > coordinates[0] else coordinates[0] - radius
        max_x = width - 1 if coordinates[0] + radius >= width else coordinates[0] + radius

        min_y = 0 if radius > coordinates[1] else coordinates[1] - radius
        max_y = height - 1 if coordinates[1] + radius >= height else coordinates[1] + radius

        for x in range(min_x, max_x):
            for y in range(min_y, max_y):
                distance_to_center = math.sqrt((x - coordinates[0]) ** 2 + (y - coordinates[1]) ** 2)
                result[y][x] = int(
                    (radius - distance_to_center) * max_value / radius) if radius > distance_to_center else 0

        return result

    def visit(self, location, timestamp, radius=0):
        """
        Makes the required changes to the map in terms of timestamp variable.
        Returns the sum of all the changes in timestamp.
        :param location: Location where the visit has been made
        :param timestamp: Current timestamp
        :param radius: Radius the agent can visit from the location
        :return: Sum of all the changes in timestamp (for easier calculation of reward)
        """
        if timestamp > 2**6 -1:
            pass
        center_x = location[0]
        center_y = location[1]
        sum = timestamp - self.map[center_x][center_y]
        self.map[center_x][center_y] += sum
        importance = (self.map[center_x][center_y] & 960) >> 6
        sum *= importance

        for i in range(2*radius):
            y = center_y - radius + i
            if y < 0:   #outside the map
                continue

            for j in range(2*radius):
                x = center_x - radius + j
                if x < 0:   # outside the map
                    continue

                if math.sqrt((y-center_y)**2 + (x-center_x)**2) > radius:   #outside the radius
                    continue

                temp_sum = timestamp - self.map[x][y]
                importance = (self.map[x][y] & 960) >> 6
                self.map[x][y] += temp_sum

                temp_sum *= importance
                sum += temp_sum

        return sum



    def _render(self):
        "Creates an image from the current map."
        return Image.fromarray(self.map, "RGB")

    def _end_state(self, map_shape, map):
        """
        Checks if whole maps is explored.
        :param map: Numpy array representing the map.
        :return: true if the map is totally explored.
        """
        flag = True
        for i in range  (0, self.map.shape[0]):
            for j in range(0, self.map.shape[1]):
                if map[i][j] & (2**6-1) == 0:
                    flag = False
        return flag
