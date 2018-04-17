#
# bondorder.py
# perform standard bond order analysis to complement the diffusion map
#
# Copyright (c) 2018 Wesley Reinhart.
# This file is part of the crayon project, released under the Modified BSD License.

from __future__ import print_function
import sys

import numpy as np
from scipy.special import sph_harm

# from crayon import nga
from crayon import parallel

def ql(rij,l):
    if len(rij.shape) == 1:
        rij = np.array([rij])
    theta = np.arctan2(rij[:,1], rij[:,0]) + np.pi
    r = np.sqrt(np.sum(rij**2.,axis=1))
    phi = np.arccos(rij[:,2]/r)
    q = 0.
    for m in range(-l,l+1):
        Y = sph_harm(m,l,theta,phi)
        qlm = np.sum(Y) / len(Y)
        q += np.abs(qlm)**2.
    q *= 4*np.pi/(2*l+1)
    return np.sqrt(q)

def computeQ(snap,averaged=True):
    q = np.zeros((snap.N,11))
    for i in range(snap.N):
        nn = snap.neighbors[i]
        nn = nn[nn!=i].flatten()
        rij = snap.wrap(snap.xyz[nn,:] - snap.xyz[i,:])
        q[i,:] = [ql(rij,2),ql(rij,3),ql(rij,4),ql(rij,5),ql(rij,6),ql(rij,7), \
                  ql(rij,8),ql(rij,9),ql(rij,10),ql(rij,11),ql(rij,12)]
    if not averaged:
        return q
    q_avg = np.zeros((snap.N,11))
    for i in range(snap.N):
        q_avg[i,:] = np.mean(q[snap.neighbors[i],:],axis=0)
    return q_avg

def computeQRange(snap):
    q_avg = computeQ(snap)
    q_range = np.max(q_avg,axis=1) - np.min(q_avg,axis=1)
    return q_range
