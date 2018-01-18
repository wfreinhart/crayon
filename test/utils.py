#
# utils.py
# provides utility functions for handling libgraphlet output
#
# Copyright (c) 2018 Wesley Reinhart.
# This file is part of the crayon project, released under the Modified BSD License.

import numpy as np

def load_uneven(filename):
    with open(filename,'r') as fid:
        lines = fid.read().split('\n')
    n = 0
    x = []
    for l in lines:
        if len(l) == 0:
            continue
        m = l.split(' ')
        x.append(m)
        if len(m) > n:
            n = len(m)
    y = np.zeros((len(x),n))
    for i in range(len(x)):
        for j in range(len(x[i])):
            y[i,j] = float( x[i][j] )
    return y
