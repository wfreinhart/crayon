#
# bondorder.py
# perform standard bond order analysis to complement the diffusion map
#
# Copyright (c) 2018 Wesley Reinhart.
# This file is part of the crayon project, released under the Modified BSD License.

from __future__ import print_function
import sys

import numpy as np

from crayon import nga
from crayon import parallel

try:
    import freud
    foundFreud = True
except:
    print('Warning: freud python module not found, bondorder module will not be available')
    foundFreud = False

class Analyzer:
    def __init__(self,quantities):
        self.quantities = quantities
        self.comm, self.size, self.rank, self.master = parallel.info()
    def guessLocality(self,snap):
        if snap.neighbors is None:
            snap.buildNeighborhoods()
        d = np.zeros(snap.N)
        for i in range(snap.N):
            dvec = snap.wrap( snap.xyz[snap.neighbors[i],:] - snap.xyz[i,:] )
            d[i] = np.median(np.linalg.norm(dvec,axis=1))
        dmed = np.median(d)
        dmad = np.median(np.abs(d-dmed))
        return dmed, dmad
    def compute(self,snap):
        dmed, dmad = self.guessLocality(snap)
        dmax = dmed + 3. * dmad
        is2D = 'z' not in snap.pbc
        box = freud.box.Box(Lx=snap.L[0],Ly=snap.L[1],Lz=snap.L[2],is2D=is2D)
        results = np.zeros((snap.N,len(self.quantities))) * np.nan
        for i, quantity in enumerate(self.quantities):
            l = int(quantity[1:])
            if quantity[0] == 'Q':
                compute = freud.order.LocalQl(box,dmax,l,0.)
                compute.computeAve(snap.xyz)
                Ql = compute.getAveQl()
                Ql[Ql!=Ql] = 0.
                results[:,i] = np.real(Ql)
            elif quantity[0] == 'W':
                compute = freud.order.LocalWl(box,dmax,l)
                compute.computeAve(snap.xyz)
                Wl = compute.getAveWl()
                Wl[Wl!=Wl] = 0.
                results[:,i] = np.real(Wl)
            else:
                raise ValueError('quantities must be of the form Ql or Wl')
        return results
    def analyzeFromFile(self,filenames,nl):
        self.filenames = filenames
        local_file_idx  = parallel.partition(range(len(self.filenames)))
        for f in local_file_idx:
            filename = self.filenames[f]
            print('rank %d of %d will process %s'%(self.rank,self.size,filename))
            # create snapshot instance and build neighborhoods
            snap = nga.Snapshot(filename,pbc='xyz',nl=nl)
            R = self.compute(snap)
            # np.savetxt(filename + '.bondorder',R,header=','.join(self.quantities))
            np.savetxt(filename + '.bondorder',R)
