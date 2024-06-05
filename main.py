import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse

from data.scripts.geometrical import get_all_second_moments
from data.scripts.parameters import *
from data.scripts.constants import *
from data.scripts.stress import *


def main(load_factor: float, should_output: bool = True):
    spar_height = SHEET_X_WIDTH + 2 * SHEET_Y_THICKNESS

    sample_z = np.arange(0, sum(WEB_LENGTHS), Z_SAMPLE_STEP)
    sample_z.reshape(sample_z.size, 1)
    sample_y = np.arange(-0.5 * spar_height, 0.5 * spar_height + Y_SAMPLE_STEP, Y_SAMPLE_STEP)
    sample_y.reshape(sample_y.size, 1)

    first_moments = get_all_first_moments(sample_z)
    second_moments = get_all_second_moments(sample_z)
    cross_sectional_areas = get_all_cross_sectional_areas(sample_z)
    internal_shears = load_factor * get_all_internal_shears(sample_z)
    internal_moments = load_factor * get_all_internal_moments(sample_z)

    internal_shear_stresses = np.abs((internal_shears * first_moments) / (second_moments * SHEET_X_THICKNESS))

    normal_stresses = get_all_internal_normal_stresses(sample_z, sample_y, second_moments, internal_moments)
    bottom_flange_normal_stresses = normal_stresses[:, normal_stresses.shape[1] - 1]
    flange_abs_stress = np.abs(bottom_flange_normal_stresses)

    # Stringer Buckling
    critical_stringer_forces = get_all_critical_stringer_forces(sample_z)
    stringer_area = STRINGER_WIDTH ** 2 - (STRINGER_WIDTH - STRINGER_THICKNESS) ** 2
    critical_stringer_stresses = critical_stringer_forces / stringer_area
    stringer_failure_points = np.greater(flange_abs_stress / 2, critical_stringer_stresses)
    stringer_failing = np.any(stringer_failure_points == True)

    # Inter Rivet Buckling
    critical_rivet_stresses = get_all_critical_rivet_stresses(sample_z)
    rivet_failure_area = np.greater(sample_z, 0.15)
    rivet_failure_points = np.logical_and(rivet_failure_area, np.greater(flange_abs_stress, critical_rivet_stresses))
    rivet_failing = np.any(rivet_failure_points == True)

    # Shear Buckling
    critical_shear_stresses = get_all_critical_shear_stresses(sample_z)
    shear_failure_points = np.greater(internal_shear_stresses, critical_shear_stresses)
    shear_failing = np.any(shear_failure_points == True)

    # Thin Sheet Buckling
    critical_thin_sheet_stresses = get_all_critical_thin_sheet_stresses(sample_z)
    thin_sheet_failure_points = np.greater(flange_abs_stress, critical_thin_sheet_stresses)
    thin_sheet_failing = np.any(thin_sheet_failure_points == True)

    # Mass
    volume = np.sum(cross_sectional_areas * Z_SAMPLE_STEP)

    num_bolts = len(BOLT_POSITIONS)
    mass_bolts = BOLT_MASS * num_bolts
    mass_nuts = NUT_MASS * num_bolts

    mass = volume * DENSITY + mass_bolts + mass_nuts

    num_holes = len(HOLE_POSITIONS)

    failing_overall = stringer_failing | rivet_failing | shear_failing | thin_sheet_failing
    if should_output:
        print("MASS:", mass, "[kg]")
        print(f"BOLTS: {num_bolts * 6}")
        print(f"HOLES: {num_holes}")
        print("FAILING:", failing_overall)
        if failing_overall:
            print("\tSTRINGER:", stringer_failing)

            print(f"\tRIVET: {rivet_failing}")
            if rivet_failing:
                print(f"\t\tFailed at: {sample_z[rivet_failure_points][0]} m")
                print(f"\t\tFail Stress: {flange_abs_stress[rivet_failure_points][0]} Pa")
                print(f"\t\tCrit Stress: {critical_rivet_stresses[rivet_failure_points][0]} Pa")

            print("\tSHEAR:", shear_failing)
            if shear_failing:
                print(f"\t\tFailed at: {sample_z[shear_failure_points][0]} m")
                print(f"\t\tFail Shear Stress: {internal_shear_stresses[shear_failure_points][0]} Pa")
                print(f"\t\tCrit Shear Stress: {critical_shear_stresses[shear_failure_points][0]} Pa")

            print("\tTHIN:", thin_sheet_failing)

    # Plotting
    if DISPLAY_PLOTS:
        internal_stress_fig, internal_stress_ax = plt.subplots()
        c = internal_stress_ax.imshow(normal_stresses.transpose(), cmap="jet", interpolation="none",
                      extent=(0, sum(WEB_LENGTHS), -spar_height / 2, spar_height / 2))
        internal_stress_fig.colorbar(c, orientation="horizontal")
        for index, hole_pos in enumerate(HOLE_POSITIONS):
            y = (2 * (index % 2) - 1) * (20 / 1000)
            hole_circle = Ellipse((hole_pos, y), width=2 * HOLE_RADIUS, height=2 * HOLE_RADIUS, color='white')
            internal_stress_ax.add_patch(hole_circle)

        line_vrange = np.linspace(-spar_height / 2, spar_height / 2)
        pos = 0
        for index in range(len(WEB_LENGTHS)):
            pos += WEB_LENGTHS[index]
            line_xrange = np.full_like(line_vrange, pos)
            internal_stress_ax.plot(line_xrange, line_vrange, color='white', alpha=0.5)

        # for bolt_pos in BOLT_POSITIONS:
        #     line_xrange = np.full_like(line_vrange, bolt_pos)
        #     internal_stress_ax.plot(line_xrange, line_vrange, color='black', alpha=0.2)

        internal_stress_ax.set_xlabel("z [m]")
        internal_stress_ax.set_ylabel("y [m]")

        stringer_fig, stringer_ax = plt.subplots()
        stringer_ax.plot(sample_z, flange_abs_stress / 2)
        stringer_ax.plot(sample_z, critical_stringer_stresses)
        stringer_ax.set_xlabel("Longitudinal Position [m]")
        stringer_ax.set_ylabel("Stress [Pa]")
        stringer_ax.grid()

        rivet_fig, rivet_ax = plt.subplots()
        rivet_ax.plot(sample_z, flange_abs_stress)
        rivet_ax.plot(sample_z, critical_rivet_stresses)
        rivet_ax.set_xlabel("Longitudinal Position [m]")
        rivet_ax.set_ylabel("Stress [Pa]")
        rivet_ax.grid()

        shear_fig, shear_ax = plt.subplots()
        shear_ax.plot(sample_z, internal_shear_stresses)
        shear_ax.plot(sample_z, critical_shear_stresses)
        shear_ax.set_xlabel("Longitudinal Position [m]")
        shear_ax.set_ylabel("Stress [Pa]")
        shear_ax.grid()

        sheet_fig, sheet_ax = plt.subplots()
        sheet_ax.plot(sample_z, flange_abs_stress)
        sheet_ax.plot(sample_z, critical_thin_sheet_stresses)
        sheet_ax.set_xlabel("Longitudinal Position [m]")
        sheet_ax.set_ylabel("Stress [Pa]")
        sheet_ax.grid()

        if SAVE_PLOTS:
            stringer_fig.savefig("data/out/stringer_fig.pdf", bbox_inches='tight')
            rivet_fig.savefig("data/out/rivet_fig.pdf", bbox_inches='tight')
            shear_fig.savefig("data/out/shear_fig.pdf", bbox_inches='tight')
            sheet_fig.savefig("data/out/sheet_fig.pdf", bbox_inches='tight')
            internal_stress_fig.savefig("data/out/internal_stress_fig.pdf", bbox_inches='tight')

        plt.show()

    return failing_overall


if __name__ == '__main__':
    det = False
    if det:
        load_factor = 1.0
        factor_step = 0.01
        failing = False

        while not failing:
            print(f"Testing Factor: {load_factor}...")
            failing = main(load_factor, False)
            load_factor += factor_step
            print("\n")

        DISPLAY_PLOTS = True
        main(load_factor - factor_step)

        print("Failed at load factor:", load_factor - factor_step)
    else:
        # DISPLAY_PLOTS = True
        # main(1)

        print(";".join(map(str, WEB_LENGTHS)))

