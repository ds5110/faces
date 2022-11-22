'''
Defining some groups of features (see the derived.md for more information on the features)
Output should be a list of the columns so the helper function (get_Xy) can appropriatly split the data
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
angle_cols = [*raw_angle_cols, *abs_angle_cols]

# ----- landmarks
x_cols, y_cols = [
    [f'gt-{axis}{i}' for i in range(68)] for axis in ['x','y']
]
landmark_cols = [*x_cols, *y_cols]

x_cols_cenrot, y_cols_cenrot = [
    [f'cenrot-{axis}{i}' for i in range(68)] for axis in ['x','y']
]
cenrot_cols = [*x_cols_cenrot, *y_cols_cenrot]

x_cols_norm, y_cols_norm = [
    [f'norm-{axis}{i}' for i in range(68)] for axis in ['x','y']
]
norm_cols = [*x_cols_norm, *y_cols_norm]

x_cols_norm_cenrot, y_cols_norm_cenrot = [
    [f'norm_cenrot-{axis}{i}' for i in range(68)] for axis in ['x','y']
]
norm_cenrot_cols = [*x_cols_norm_cenrot, *y_cols_norm_cenrot]

# ----- symmetrical differences
x_sym_diff, y_sym_diff = [
    [f'sym_diff-{axis}{i}' for [i, j] in h_syms] for axis in ['x','y']
]
sym_diff = [*x_sym_diff, *y_sym_diff]
x_norm_cenrot_sym_diff, y_norm_cenrot_sym_diff = [
    [f'norm_cenrot_sym_diff-{axis}{i}' for [i, j] in h_syms] for axis in ['x','y']
]
norm_cenrot_sym_diff = [*x_norm_cenrot_sym_diff, *y_norm_cenrot_sym_diff]