# -*- coding: utf-8 -*-
# Copyright © 2022 Thales. All Rights Reserved.
# NOTICE: This file is subject to the license agreement defined in file 'LICENSE', which is part of
# this source code package.

from src.kesslergame import KesslerController
from typing import Dict, Tuple
import random
import math
from typing import Dict, Tuple
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import numpy as np
import heapq
import time
import os
import bisect
from functools import lru_cache
import matplotlib.pyplot as plt

class Smith(KesslerController):
    def __init__(self):
        """
        Any variables or initialization desired for the controller can be set up here
        """
        self.current_timestep = 0

    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool, bool]:
        """
        Method processed each time step by this controller to determine what control actions to take

        Arguments:
            ship_state (dict): contains state information for your own ship
            game_state (dict): contains state information for all objects in the game

        Returns:
            float: thrust control value
            float: turn-rate control value
            bool: fire control value. Shoots if true
            bool: mine deployment control value. Lays mine if true
        """
        self.current_timestep += 1

        thrust = 0
        turn_rate = 0
        fire = False
        drop_mine = False

        if self.current_timestep == 1:
            thrust, turn_rate, fire, drop_mine = 0, 0, 0, True
        elif 10 < self.current_timestep < 30:
            thrust, turn_rate, fire, drop_mine = 280, 0, 0, False

        return thrust, turn_rate, fire, drop_mine

    @property
    def name(self) -> str:
        """
        Simple property used for naming controllers such that it can be displayed in the graphics engine

        Returns:
            str: name of this controller
        """
        return "Smith"
