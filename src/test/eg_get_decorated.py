# -*- coding: utf-8 -*-

"""
This shows how to get the 'decorated' metadata DataFrame.

@author: jhautala
"""

from util.local_cache import cache
from util.model import nose_i

df = cache.get_meta('decorated')

cols = [col for col in df.columns if col.startswith(f'norm_cenrot-')]
df[cols].describe()

#-- these should all be zeros
df[[f'norm_cenrot-{dim}{nose_i}' for dim in ['x','y']]].describe()

#-- these might be interesting?
df[[f'cenrot-{dim}{nose_i}' for dim in ['x','y']]].describe()