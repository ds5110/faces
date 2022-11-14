#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 20:08:59 2022

@author: jhautala
"""
import matplotlib.pyplot as plt

# project
from util.local_cache import cache
from util.pre import to_deg

savefig = True

df = cache.get_meta('decorated')

def plot(use_abs=False,savefig=False):
    label_pre = 'absolute ' if use_abs else ''
    col_suff = '_abs' if use_abs else ''
    tilted = df['tilted'] == 1
    turned = df['turned'] == 1
    both = tilted & turned
    neither = ~(tilted | turned)
    for (mask, color, desc) in [
            (neither, 'tab:gray', 'neither'),
            (tilted, 'tab:blue', 'tilted'),
            (turned, 'tab:red', 'turned'),
            (both, 'tab:purple', 'both'),
    ]:
        tmp = df[mask]
        plt.scatter(
            tmp[f'yaw{col_suff}'] * to_deg,
            tmp[f'roll{col_suff}'] * to_deg,
            s=10,
            c=color,
            label=desc,
        )
    plt.title('estimated angular offsets')
    plt.xlabel(f'{label_pre}yaw (in degrees)')
    plt.ylabel(f'{label_pre}roll (in degrees)')
    plt.legend(
        bbox_to_anchor=(1.02, 1),
        loc='upper left',
        borderaxespad=0.,
    )
    plt.tight_layout()
    if savefig:
        plt.savefig(f'figs/roll_yaw{col_suff}.png', dpi=300, bbox_inches='tight')
    plt.show()

plot(savefig=savefig)
plot(use_abs=True,savefig=savefig)