#
# dmap.py
# wraps PyDMap diffusion map calculations for easy scripting
#
# Copyright (c) 2018 Wesley Reinhart.
# This file is part of the crayon project, released under the Modified BSD License.

from __future__ import print_function
import pickle

import numpy as np

from crayon import parallel

try:
    import PyDMap
except:
    raise RuntimeError('dmap submodule requires PyDMap python module')

class DMap:
    def __init__(self):
        self._cpp = PyDMap.DMap()
        self.alpha = 1.0
        self.num_evec = 4
        self.epsilon = None
        self.evals = None
        self.evecs = None
        self.evecs_ny = None
        self.color_coords = None
    def build(self,dists,landmarks=None,
              valid_cols=None,
              valid_rows=None):
        if valid_rows is None:
            valid_rows = np.arange(dists.shape[0])
        if valid_cols is None:
            valid_cols = np.arange(dists.shape[1])
        alpha_dists = np.power(dists,self.alpha)
        if landmarks is None:
            D = alpha_dists
            D = D[valid_rows,:]
            D = D[:,valid_cols]
        else:
            D = alpha_dists[landmarks,:]
            D = D[:,valid_cols]
            L = alpha_dists[valid_rows,:]
            L = L[:,valid_cols]
        # compute landmark manifold
        self._cpp.set_dists(D)
        self._cpp.set_num_evec(self.num_evec)
        if self.epsilon is None:
            self.epsilon = np.median(alpha_dists)
        self._cpp.set_epsilon(self.epsilon)
        self._cpp.compute()
        self.evals = np.asarray(self._cpp.get_eval())
        self.evecs = np.asarray(self._cpp.get_evec())
        # embed remaining graphs using landmarks
        if landmarks is None:
            self.evecs_ny = None
        else:
            self.evecs_ny = np.asarray(PyDMap.nystrom(self._cpp,L))
        # backfill for invalid graphs
        evecs = np.zeros((dists.shape[1],self.num_evec))*np.nan
        evecs[valid_cols,:] = np.array(self.evecs)
        self.evecs = evecs
        evecs_ny = np.zeros((dists.shape[0],self.num_evec))*np.nan
        evecs_ny[valid_rows,:] = np.array(self.evecs_ny)
        self.evecs_ny = evecs_ny
        # construct uniformly coordinates for mapping to RGB space
        if self.evecs_ny is None:
            R = self.evecs
        else:
            R = self.evecs_ny
        self.color_coords = color.rankOrderTransform(R)
        # first eigenvector is always trivial
        self.color_coords[:,0] = 0.5
        # catch nan rows
        nan_idx = np.argwhere(np.isnan(self.color_coords[:,-1])).flatten()
        self.color_coords[nan_idx,:] = 1.
    def write(self,prefix='',binary=False):
        if binary:
            buff = {'evals': self.evals,
                    'evecs': self.evecs,
                    'evecs_ny': self.evecs_ny,
                    'color_coords': self.color_coords}
            with open('%sdmap.bin'%prefix,'wb') as fid:
                pickle.dump(buff,fid)
        else:
            np.savetxt('%sevals.dat'%prefix,self.evals)
            np.savetxt('%sevecs-lm.dat'%prefix,self.evecs)
            if self.evecs_ny is not None:
                np.savetxt('%sevecs-ny.dat'%prefix,self.evecs_ny)
            if self.color_coords is not None:
                np.savetxt('%scolor-coords.dat'%prefix,self.color_coords)
    def uncorrelatedTriplets(self):
        # first determine least correlated eigenvectors
        X = np.abs( np.corrcoef(np.transpose(self.color_coords[:,1:])) )
        Y = np.zeros(X.shape[1])
        coms = []
        for i in range(X.shape[1]):
            com = tuple( np.sort( [i+1] + list(np.argsort(X[:,i])[:2]+1) ) )
            if com not in coms:
                coms.append(com)
            Y[i] = np.sum(X[i,np.asarray(com)-1])
        best = np.argwhere(Y[i] == np.min(Y[i]))
        return coms, best
