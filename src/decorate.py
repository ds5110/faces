# -*- coding: utf-8 -*-

"""
This script decorates the original metadata DataFrame, adding some
geometric information derived from landmark data.

@author: jhautala
"""

# intra-project
from util.local_cache import cache
from util.pre import add_derived


if __name__ == '__main__':
    # load the labels data
    decorated = add_derived(cache)
    cache.save_meta(decorated, 'decorated')
