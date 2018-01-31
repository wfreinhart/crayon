#
# dmap.py
# wraps PyDMap diffusion map calculations for easy scripting
#
# Copyright (c) 2018 Wesley Reinhart.
# This file is part of the crayon project, released under the Modified BSD License.

from __future__ import print_function
import pickle

import numpy as np

try:
    import PyDMap
except:
    raise RuntimeError('dmap submodule requires PyDMap python module')

class DMap:
    def __init__(self):
        self._cpp = PyDMap.DMap()
        self.set_params()
        self.evals = None
        self.evecs = None
        self.evecs_ny = None
        self.color_coords = None
    def set_params(self,num_evec=6,epsilon=None):
        self.num_evec = num_evec
        self.epsilon = epsilon
    def build(self,L,landmarks=None):
        if landmarks is None:
            D = L
        else:
            D = L[landmarks,:]
        # compute landmark manifold
        self._cpp.set_dists(D)
        self._cpp.set_num_evec(self.num_evec)
        if self.epsilon is None:
            self.epsilon = np.median(L)
        self._cpp.set_epsilon(self.epsilon)
        self._cpp.compute()
        self.evals = np.asarray(self._cpp.get_eval())
        self.evecs = np.asarray(self._cpp.get_evec())
        # embed remaining graphs using landmarks
        if landmarks is None:
            self.evecs_ny = None
        else:
            self.evecs_ny = np.asarray(PyDMap.nystrom(self._cpp,L))
        self.transform()
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
    def transform(self,prefix=''):
        # transform data to uniform distribution
        if self.evecs_ny is None:
            R = self.evecs
        else:
            R = self.evecs_ny
        self.color_coords = np.zeros(R.shape)
        # first eigenvector is always trivial
        self.color_coords[:,0] = 0.5
        # transform each remaining eigenvector to yield a uniform distribution
        for i in range(1,R.shape[1]):
            r = R[:,i]
            x = np.linspace(np.min(r),np.max(r),np.round(np.sqrt(len(r))))
            hy, hx = np.histogram(r, bins=x, normed=True)
            c = np.cumsum(hy) / np.sum(hy)
            self.color_coords[:,i] = np.interp(r,0.5*(x[:-1]+x[1:]),c)
