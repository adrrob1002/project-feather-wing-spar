import math
from functools import cache

import numpy as np
import matplotlib.pyplot as plt

from data.scripts.constants import *
from data.scripts.parameters import HOLE_POSITIONS


def get_flange_second_moment_of_area() -> float:
    b = SHEET_Y_WIDTH  # [m]
    h = SHEET_Y_THICKNESS  # [m]
    a = SHEET_X_WIDTH / 2  # [m]

    return ((b * h ** 3) / 12) + b * h * (a + h / 2) ** 2  # [m4]


def get_stringer_second_moment_of_area(y: float) -> float:
    t = STRINGER_THICKNESS
    b = STRINGER_WIDTH

    vertical_part = (t * (b - t) ** 3) / 12 + t * (b - t) * (y - t - (b - t) / 2) ** 2  # [m4]
    horizontal_part = (b * t ** 3) / 12 + b * t * (y - t / 2) ** 2  # [m4]

    return vertical_part + horizontal_part  # [m4]


def get_web_second_moment_of_area_at(z: float) -> float:
    horizontal_distance_to_hole = abs(get_vector_to_closest_hole(z)[0])
    hole_height = 0

    if horizontal_distance_to_hole < HOLE_RADIUS:
        hole_height = 2 * math.sqrt(HOLE_RADIUS ** 2 - horizontal_distance_to_hole ** 2)

    # print(hole_height)

    return (SHEET_X_THICKNESS * (SHEET_X_WIDTH - hole_height) ** 3) / 12


@cache
def get_second_moment_of_area_at(z: float) -> float:
    stringer_y = SHEET_X_WIDTH / 2

    web = get_web_second_moment_of_area_at(z)
    stringer = 4 * get_stringer_second_moment_of_area(stringer_y)
    flanges = 2 * get_flange_second_moment_of_area()

    return web + stringer + flanges


get_all_second_moments = np.vectorize(get_second_moment_of_area_at)


def get_vector_to_closest_hole(z: float, y: float = 0) -> np.ndarray:
    if len(HOLE_POSITIONS) == 0:
        return np.full(2, np.inf)

    hole_positions = np.asarray(HOLE_POSITIONS)
    closest_hole_index = (np.abs(hole_positions - z)).argmin()

    closest_hole_position = np.array([hole_positions[closest_hole_index], 0])
    current_position = np.array([z, y])

    return closest_hole_position - current_position


@cache
def get_cross_sectional_area_at(z: float) -> float:
    horizontal_bar_area = SHEET_Y_THICKNESS * SHEET_Y_WIDTH  # [m2]
    vertical_stringer_area = STRINGER_THICKNESS * (STRINGER_WIDTH - STRINGER_THICKNESS)  # [m2]
    horizontal_stringer_area = STRINGER_WIDTH * STRINGER_THICKNESS  # [m2]
    web_sheet_area = SHEET_X_THICKNESS * SHEET_X_WIDTH  # [m2]

    horizontal_distance_to_hole = abs(get_vector_to_closest_hole(z)[0])
    hole_height = 0

    if horizontal_distance_to_hole < HOLE_RADIUS:
        hole_height = 2 * math.sqrt(HOLE_RADIUS ** 2 - horizontal_distance_to_hole ** 2)

    hole_area = SHEET_X_THICKNESS * hole_height
    web_sheet_area -= hole_area
    stringer_area = vertical_stringer_area + horizontal_stringer_area

    return 2 * horizontal_bar_area + 4 * stringer_area + web_sheet_area  # [m2]


def get_first_moment_of_area_at(z: float) -> float:
    horizontal_bar_area = SHEET_Y_THICKNESS * SHEET_Y_WIDTH  # [m2]
    vertical_stringer_area = STRINGER_THICKNESS * (STRINGER_WIDTH - STRINGER_THICKNESS)  # [m2]
    horizontal_stringer_area = STRINGER_WIDTH * STRINGER_THICKNESS  # [m2]
    web_sheet_area = SHEET_X_THICKNESS * (SHEET_X_WIDTH / 2)  # [m2]

    horizontal_distance_to_hole = abs(get_vector_to_closest_hole(z)[0])
    hole_height = 0

    if horizontal_distance_to_hole < HOLE_RADIUS:
        hole_height = 2 * math.sqrt(HOLE_RADIUS ** 2 - horizontal_distance_to_hole ** 2)

    hole_area = SHEET_X_THICKNESS * (hole_height / 2)

    horizontal_bar_distance = SHEET_X_WIDTH / 2 + SHEET_Y_THICKNESS / 2  # [m]
    vertical_stringer_distance = SHEET_X_WIDTH / 2 - STRINGER_THICKNESS - (STRINGER_WIDTH - STRINGER_THICKNESS) / 2
    horizontal_stringer_distance = SHEET_X_WIDTH / 2 - STRINGER_THICKNESS / 2
    web_sheet_distance = SHEET_X_WIDTH / 4
    hole_distance = hole_height / 4

    hbar = horizontal_bar_area * horizontal_bar_distance
    vstringer = vertical_stringer_area * vertical_stringer_distance
    hstringer = horizontal_stringer_area * horizontal_stringer_distance
    web = web_sheet_area * web_sheet_distance
    hole = hole_area * hole_distance

    return hbar + 2 * vstringer + 2 * hstringer + web - hole


get_all_first_moments = np.vectorize(get_first_moment_of_area_at)
