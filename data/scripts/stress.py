import math

import numpy as np

from data.scripts.constants import *
from data.scripts.geometrical import *
from data.scripts.parameters import *


def get_internal_shear_at(z: float):
    force = 0

    for force_info in APPLIED_FORCES:
        force_z, force_f = force_info[0], force_info[1]

        if z >= force_z:
            force += force_f

    return force


def get_internal_moment_at(z: float) -> float:
    moment = 0

    for moment_info in APPLIED_MOMENTS:
        if z >= moment_info[0]:
            moment += moment_info[1]

    for force_info in APPLIED_FORCES:
        force_z, force_f = force_info[0], force_info[1]

        if z >= force_z:
            distance = abs(force_z - z)
            moment += distance * force_f

    return moment


def get_all_internal_normal_stresses(z: np.ndarray, y: np.ndarray, second_moments: np.ndarray,
                                     internal_moments: np.ndarray) -> np.ndarray:
    # sigma_z = (M_z * y) / I_x
    multiplier = (internal_moments / second_moments).reshape(z.size, 1)
    y_positions = y.reshape(1, y.size)

    return np.matmul(multiplier, y_positions)

    # normal_stresses = np.zeros((z.size, y.size), dtype=np.float64)
    #
    # for z_index in range(z.size):
    #     for y_index in range(y.size):
    #         z_pos = float(z[z_index])
    #         y_pos = float(y[y_index])
    #         result = get_concentrated_stress_at(z_pos, y_pos, sigma_inf[z_index, y_index])
    #         normal_stresses[z_index, y_index] = result
    #
    # return normal_stresses
    # return sigma_inf


def ease_in_out_quad(x: float) -> float:
    return 2 * x * x if x < 0.5 else 1 - math.pow(-2 * x + 2, 2) / 2


def get_concentrated_stress_at(z: float, y: float, sigma_inf: float) -> float:
    if len(HOLE_POSITIONS) == 0:
        return sigma_inf

    position = np.array([z, y])

    sigma_overall = 0

    for hole_z in HOLE_POSITIONS:
        hole_position = np.array([hole_z, 0])
        offset = position - hole_position

        theta = math.atan2(offset[1], offset[0])
        sigma_inf = sigma_inf / len(HOLE_POSITIONS)

        a = HOLE_RADIUS
        r = np.linalg.norm(offset)

        if r > HOLE_RADIUS:
            a_to_r = a / r

            # https://www.fracturemechanics.org/hole.html
            sigma_rr = (sigma_inf / 2) * (1 - (a_to_r ** 2)) + (sigma_inf / 2) * (
                    1 - (4 * a_to_r ** 2) + 3 * (a_to_r ** 4)) * math.cos(2 * theta)
            sigma_tt = ((sigma_inf / 2) * (1 + (a_to_r ** 2)) - (sigma_inf / 2) * (1 + (3 * a_to_r ** 4)) * math.cos(
                2 * theta))

            sigma_overall += (sigma_rr + sigma_tt)

    return sigma_overall


def get_critical_stringer_buckling_force_at(z: float) -> float:
    second_moment = get_stringer_second_moment_of_area(0)  # [m4]
    length = STRINGER_LENGTH  # [m]
    e_modulus = E_MODULUS  # [Pa]

    return (STRINGER_END_FIXITY_COEFFICIENT * (math.pi ** 2) * e_modulus * second_moment) / (length ** 2)  # [N]


def get_critical_rivet_buckling_stress_at(z: float) -> float:
    bolt_start = 0.15
    bolt_end = 2.25

    if z >= BOLT_POSITIONS[-1]:
        bolt_start = BOLT_POSITIONS[-1]

    if z < BOLT_POSITIONS[0]:
        bolt_end = BOLT_POSITIONS[0]

    if BOLT_POSITIONS[0] <= z < BOLT_POSITIONS[-1]:
        bolt_start = [pos for pos in BOLT_POSITIONS if pos <= z][-1]  # [mm]
        bolt_end = [pos for pos in BOLT_POSITIONS if pos > z][0]  # [mm]

    spacing = (bolt_end - bolt_start)  # [m]
    t = SHEET_Y_THICKNESS + STRINGER_THICKNESS  # [m]
    modulus = E_MODULUS  # [Pa]

    return 0.9 * RIVET_CONSTANT * modulus * (t / spacing) ** 2


def get_critical_shear_buckling_stress_at(z: float) -> float:
    web_segment_index = 0
    total_length = 0

    for index in range(len(WEB_LENGTHS)):
        segment_length = WEB_LENGTHS[index]

        if total_length <= z < total_length + segment_length:
            web_segment_index = index

        total_length += segment_length


    a = WEB_LENGTHS[web_segment_index]

    if web_segment_index == 0:
        a -= 0.15

    b = SHEET_X_WIDTH
    ab_ratio = a / b

    shear_buckling_coefficient = np.interp(ab_ratio, SHEAR_BUCKLING_COEFFICIENT_RATIOS, SHEAR_BUCKLING_COEFFICIENTS)

    t = SHEET_X_THICKNESS  # [m]
    w = SHEET_X_WIDTH  # [m]
    modulus = E_MODULUS  # [Pa]

    return shear_buckling_coefficient * modulus * (t / w) ** 2  # [Pa]


def get_critical_thin_sheet_buckling_stress_at(z: float) -> float:
    # sigma_cr = (k * pi^2 * E) / (12 * (1 - nu^2) * (b / t)^2
    bolt_start = 0.15
    bolt_end = 2.25

    if z >= BOLT_POSITIONS[-1]:
        bolt_start = BOLT_POSITIONS[-1]

    if z < BOLT_POSITIONS[0]:
        bolt_end = BOLT_POSITIONS[0]

    if BOLT_POSITIONS[0] <= z < BOLT_POSITIONS[-1]:
        bolt_start = [pos for pos in BOLT_POSITIONS if pos <= z][-1]  # [mm]
        bolt_end = [pos for pos in BOLT_POSITIONS if pos > z][0]  # [mm]

    a = (bolt_end - bolt_start)  # [m]
    b = HOLE_INNER_SPACING
    ab_ratio = a / b
    k = np.interp(ab_ratio, THIN_SHEET_BUCKLING_COEFFICIENT_RATIOS, THIN_SHEET_BUCKLING_COEFFICIENTS)

    t = SHEET_Y_THICKNESS
    e_modulus = E_MODULUS

    return k * e_modulus * ((t / b) ** 2)


get_all_internal_shears = np.vectorize(get_internal_shear_at)
get_all_internal_moments = np.vectorize(get_internal_moment_at)
get_all_cross_sectional_areas = np.vectorize(get_cross_sectional_area_at)
get_all_critical_stringer_forces = np.vectorize(get_critical_stringer_buckling_force_at)
get_all_critical_rivet_stresses = np.vectorize(get_critical_rivet_buckling_stress_at)
get_all_critical_shear_stresses = np.vectorize(get_critical_shear_buckling_stress_at)
get_all_critical_thin_sheet_stresses = np.vectorize(get_critical_thin_sheet_buckling_stress_at)
