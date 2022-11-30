#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 20:08:59 2022

@author: jhautala
"""

import matplotlib.pyplot as plt

# project
from util import meta_cache
from util.pre import to_deg


def plot(df, use_abs=False, save_fig=False):
    label_pre = 'absolute ' if use_abs else ''
    col_suff = '_abs' if use_abs else ''
    tilt = df['tilted'] == 1
    turn = df['turned'] == 1
    tilted = tilt & ~turn
    turned = turn & ~tilt
    both = tilt & turn
    neither = ~tilt & ~turn
    for (mask, color, desc) in [
            (neither, 'black', 'neither'),
            (tilted, 'tab:blue', 'tilted'),
            (turned, 'tab:red', 'turned'),
            (both, 'tab:purple', 'both'),
    ]:
        tmp = df[mask]
        plt.scatter(
            tmp[f'yaw{col_suff}'] * to_deg,
            tmp[f'roll{col_suff}'] * to_deg,
            s=10,
            # c=color,
            marker='x' if desc == 'both' else 'o',
            label=desc,
            edgecolors=None if desc == 'both' else color,
            facecolors='none' if desc == 'neither' else color,
            # marker='$O$' if desc == 'neither' else 'o',
        )
    plt.title('estimated angular offsets')
    plt.xlabel(f'{label_pre}yaw (in degrees)')
    plt.ylabel(f'{label_pre}roll (in degrees)')
    plt.legend()
    
    #-- fix the legend
    ax = plt.gca()
    intended = ['tilted', 'turned', 'both', 'neither']
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(l) for l in intended]
    ax.legend(
        [handles[i] for i in order], [labels[i] for i in order],
        bbox_to_anchor=(1.02, 1),
        loc='upper left',
        borderaxespad=0.,
    )
    plt.tight_layout()
    if save_fig:
        plt.savefig(f'figs/roll_yaw{col_suff}.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    save_fig = False
    df = meta_cache.get_meta('baby')
    plot(df, save_fig=save_fig)
    plot(df, use_abs=True, save_fig=save_fig)


if __name__ == '__main__':
    main()