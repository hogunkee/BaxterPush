import numpy as np
from robosuite.models.robots.robot import Robot
from robosuite.utils.mjcf_utils import xml_path_completion, array_to_string


class Swimmer(Robot):
    """Baxter is a hunky bimanual robot designed by Rethink Robotics."""

    def __init__(self):
        super().__init__(xml_path_completion("robots/swimmer/robot.xml"))
