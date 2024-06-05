# SHEET X (WEB SHEET)
SHEET_X_LENGTH = 1.5  # [m]
SHEET_X_WIDTH = 0.1484  # [m]
SHEET_X_THICKNESS = 0.8 / 1000  # [m]

SHEAR_BUCKLING_COEFFICIENT_RATIOS = [1, 1.5, 2, 3, 4, 5, 6]
SHEAR_BUCKLING_COEFFICIENTS = [11, 7.8, 7, 6, 5.6, 5.2, 5.2]

# SHEET Y (TOP/BOTTOM SHEET)
SHEET_Y_LENGTH = 1.5  # [m]
SHEET_Y_WIDTH = 40 / 1000  # [m]
SHEET_Y_THICKNESS = 0.8 / 1000  # [m]

THIN_SHEET_BUCKLING_COEFFICIENT_RATIOS = [0.7, 1, 1.2, 1.5, 2, 3, 4, 5, 6]  # [-]
THIN_SHEET_BUCKLING_COEFFICIENTS = [9, 6, 5.4, 4.8, 4.3, 3.8, 3.6, 3.6, 3.6]  # [-]

# STRINGER
STRINGER_LENGTH = 1.5  # [m]
STRINGER_WIDTH = 20 / 1000  # [m]
STRINGER_THICKNESS = 1.5 / 1000  # [m]
STRINGER_END_FIXITY_COEFFICIENT = 7.5  # [-]

# BOLTS & NUTS
MAX_BOLTS = 60
MAX_NUTS = 60
BOLT_MASS = 1.39 / 1000  # [kg]
NUT_MASS = 0.94 / 1000  # [kg]
RIVET_CONSTANT = 3.5  # [-]

# MATERIAL
ULTIMATE_STRESS = 483 * 10 ** 6  # [Pa]
YIELD_STRESS = 345 * 10 ** 6  # [Pa]
E_MODULUS = 71700 * 10 ** 6  # [Pa]
DENSITY = 2780  # [kg/m3]

# HOLES
HOLE_RADIUS = 20 / 1000  # [m]

# APPLIED FORCES
APPLIED_FORCES = [(0, -1595.1), (0.72, 1141.26), (0.72 + 1.2, 508.74), (2.25, -54.94)]  # [(m, N)]
APPLIED_MOMENTS = [(0, 1674.8)]  # [(m, Nm)]

# SIMULATION SETTINGS
SAVE_PLOTS = True
DISPLAY_PLOTS = False

Z_SAMPLE_STEP = 1 / 1000  # [m]
Y_SAMPLE_STEP = 1 / 100  # [m]