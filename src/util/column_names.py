'''
Here we define some groups of features (see the derived.md for more information on the features)
Output should be a list of the columns so the helper function (get_Xy) can appropriatly split the data

NOTE: The normalized columns are normalized per the extent of the minimum bounding
      box around landmarks. It may be worth considering also/instead normalizing
      per 'boxsize' (i.e. the area of the minimum bounding box).
'''

from util.model import h_syms


# ----- 'subjective' stuff (should be left to each model?)
target = 'baby'
main_predictors = [
    'boxratio',
    'interoc',
    'interoc_norm',
    'boxsize',
    'boxsize/interoc',
]


# ----- 'challenging' category labels
cat_cols = ['turned', 'occluded', 'tilted', 'expressive']


# ----- metadata columns
sz_cols = ['width', 'height', 'cenrot_width', 'cenrot_height']
meta_cols = ['image-set', 'filename', 'partition', 'subpartition']
face_dims = ['face_width', 'face_height', 'boxratio', 'boxsize']
sz_cols = [
    'width',
    'height',
    'cenrot_width',
    'cenrot_height',
    'scale',
    'center_w',
    'center_h',
]
meta_cols = [
    'image-set',
    'filename',
    'image_name',
    'partition',
    'subpartition',
]

# ----- higher level columns
interaction_cols = ['boxsize/interoc', 'interoc_norm']

# - estimated angular offsets
raw_angle_cols = ['yaw', 'roll']
abs_angle_cols = [f'{col}_abs' for col in raw_angle_cols]
all_angle_cols = [*raw_angle_cols, *abs_angle_cols]

# ----- landmarks
# - baby landmarks
x_cols, y_cols = [
    [f'gt-{axis}{i}' for i in range(68)] for axis in ['x','y']
]
landmark_cols = [*x_cols, *y_cols]

# - centered/rotated
x_cols_cenrot, y_cols_cenrot = [
    [f'cenrot-{axis}{i}' for i in range(68)] for axis in ['x','y']
]
cenrot_cols = [*x_cols_cenrot, *y_cols_cenrot]

# normalized
x_cols_norm, y_cols_norm = [
    [f'norm-{axis}{i}' for i in range(68)] for axis in ['x','y']
]
norm_cols = [*x_cols_norm, *y_cols_norm]

# centered/rotated and normalized per extent
x_cols_norm_cenrot, y_cols_norm_cenrot = [
    [f'norm_cenrot-{axis}{i}' for i in range(68)] for axis in ['x','y']
]
norm_cenrot_cols = [*x_cols_norm_cenrot, *y_cols_norm_cenrot]
all_landmark_cols = [
    *landmark_cols,
    *cenrot_cols,
    *norm_cols,
    *norm_cenrot_cols,
]

# - adult landmarks
alt_x_cols, alt_y_cols = [[f'original_{i}_{axis}' for i in range(68)] for axis in ['x', 'y']]
alt_cols = [*alt_x_cols, *alt_y_cols]

# - merged landmmarks
merged_cols = []
for cols in [[f'{axis}{i}' for i in range(68)] for axis in ['x', 'y']]:
    merged_cols.extend(cols)

# ----- symmetrical differences
x_sym_diff, y_sym_diff = [
    [f'sym_diff-{axis}{i}' for [i, j] in h_syms] for axis in ['x','y']
]
sym_diff = [*x_sym_diff, *y_sym_diff]
x_norm_sym_diff, y_norm_sym_diff = [
    [f'norm_sym_diff-{axis}{i}' for [i, j] in h_syms] for axis in ['x','y']
]
norm_sym_diff = [*x_norm_sym_diff, *y_norm_sym_diff]
x_norm_cenrot_sym_diff, y_norm_cenrot_sym_diff = [
    [f'norm_cenrot_sym_diff-{axis}{i}' for [i, j] in h_syms] for axis in ['x','y']
]
norm_cenrot_sym_diff = [*x_norm_cenrot_sym_diff, *y_norm_cenrot_sym_diff]
all_sym_diff = [*sym_diff, *norm_sym_diff, *norm_cenrot_sym_diff]
