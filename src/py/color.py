#
# color.py
# assign colors to particles based on position in transformed manifold
#
# Copyright (c) 2018 Wesley Reinhart.
# This file is part of the crayon project, released under the Modified BSD License.

from __future__ import print_function

import numpy as np

def rotate(coords,axis,turns):
    theta = int(turns) * 0.5 * np.pi
    R = []
    R.append( np.array([[1, 0,              0],
                        [0, np.cos(theta), -np.sin(theta)],
                        [0, np.sin(theta),  np.cos(theta)]]) )
    R.append( np.array([[ np.cos(theta), 0, np.sin(theta)],
                        [ 0,             1, 0],
                        [-np.sin(theta), 0, np.cos(theta)]]) )
    R.append( np.array([[np.cos(theta), -np.sin(theta), 0],
                        [np.sin(theta),  np.cos(theta), 0],
                        [0,              0,             1]]) )
    t = 0.5*np.ones(3)
    return t+np.matmul(coords-t,R[axis])

def rankTransform(R):
    # transform data to uniform distribution
    coords = np.zeros(R.shape)*np.nan
    # transform each remaining eigenvector to yield a uniform distribution
    for i in range(R.shape[1]):
        r = R[:,i]
        idx = np.argsort(r[r==r])
        rs = r[r==r][idx]
        x = np.linspace(0.,1.,len(rs))
        coords[:,i] = np.interp(r,rs,x)
    nan_idx = np.argwhere(np.isnan(coords[:,-1])).flatten()
    coords[nan_idx,:] = 1.
    return coords
