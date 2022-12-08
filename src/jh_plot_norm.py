#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TODO: resolve image access for non-infant

@author: jhautala
"""

from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# internal
from util import meta_cache
from util.pre import rotate, to_deg
from util.plot import plot_image
from util.column_names import x_cols_norm_cenrot, y_cols_norm_cenrot


save_fig = False

# obtain DataFrame
df = meta_cache.get_meta()

# create multiplot
fig, axs = plt.subplots(
    nrows=2,
    ncols=3,
    figsize=(12, 8),
)

for i, image_name in enumerate([
        'ads/1072.Still005.jpg', # 689
        'helen/testset/3045921676_1.jpg' # 1
]):
    ax1 = axs[i][0]
    ax2 = axs[i][1]
    ax3 = axs[i][2]
    
    row_id = df[df['image_name'] == image_name].index.values[0]
    anno = meta_cache.get_image(row_id)
    if i == 1:
        # NOTE: this is the hack to get adult face
        image_file = Path(f'data/{image_name}')
        anno.im_fun = lambda: Image.open(image_file)
    
    # plot raw
    plot_image(
        anno,
        cross=False,
        ax=ax1,
    )
    ax1.title.set_text(image_name + ' \n raw image data')
    
    rotated = rotate(anno)
    plot_image(
        rotated,
        annotate='spline',
        cross=True,
        grayscale='bone',
        ax=ax2,
    )
    yaw = df.iloc[row_id]['yaw'] * to_deg
    ax2.title.set_text(
        f'rotated and centered \n estimated yaw: {yaw:.2f}' + '$^{\circ}$'
    )
    
    # plot geometry without image, to verify coords
    # NOTE: this is a hack to plot normalized coords
    #       we probably shouldn't mutate the model
    rotated.set_coords(np.stack(
        [
            df[cols].loc[row_id,:].values
            for cols in [x_cols_norm_cenrot, y_cols_norm_cenrot]],
        1
    ))
    plot_image(
        rotated,
        annotate='splinenum',
        skip_img=True,
        ax=ax3,
    )
    ax3.title.set_text('normalized landmarks')
    ax3.set_xlim(-1, 1)
    ax3.set_ylim(1, -1)
    
plt.tight_layout()
if save_fig:
    plt.savefig(
        'figs/normalized_landmarks.png',
        dpi=300,
        bbox_inches='tight'
    )
plt.show()
    