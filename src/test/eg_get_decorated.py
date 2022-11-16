# -*- coding: utf-8 -*-

"""
This shows how to get the 'decorated' metadata DataFrame.

@author: jhautala
"""

from util.local_cache import cache
from util.model import nose_i
import matplotlib.pyplot as plt

df = cache.get_meta('decorated')

cols = [col for col in df.columns if col.startswith(f'norm_cenrot-')]
df[cols].describe()

#-- these should all be zeros
df[[f'norm_cenrot-{dim}{nose_i}' for dim in ['x','y']]].describe()

#-- these might be interesting?
df[[f'cenrot-{dim}{nose_i}' for dim in ['x','y']]].describe()

# Displays the histgram of box ratio. It's centered around 1.28 as suggested by Doctor Wang's paper
plt.hist(df['boxratio'])
plt.title('Box Ratio of Infant Faces')
# plt.savefig('boxratio_hist')
plt.show()

plt.hist(df['interoc_norm'])
plt.title('Interoc_Norm of Infant Faces')
# plt.savefig('interoc_norm_hist')
plt.show()