# -*- coding: utf-8 -*-

"""
This script decorates the original metadata DataFrame, adding some
geometric information derived from landmark data.

@author: jhautala
"""

# intra-project
from util import cache, alt
from util.pre import add_derived


if __name__ == '__main__':
    for c in [cache, alt]:
        decorated = add_derived(c)
        c.save_meta(decorated, 'decorated')
