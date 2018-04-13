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

try:
    import freud
    foundFreud = True
except:
    print('Warning: freud python module not found, bondorder module will not be available')
    foundFreud = False

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

class AnalyzerFreud:
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
    def compute(self,snap,dmax=None):
        if dmax is None:
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
    # def analyzeFromFile(self,filenames,nl):
    #     self.filenames = filenames
    #     local_file_idx  = parallel.partition(range(len(self.filenames)))
    #     for f in local_file_idx:
    #         filename = self.filenames[f]
    #         print('rank %d of %d will process %s'%(self.rank,self.size,filename))
    #         # create snapshot instance and build neighborhoods
    #         snap = nga.Snapshot(filename,pbc='xyz',nl=nl)
    #         R = self.compute(snap)
    #         # np.savetxt(filename + '.bondorder',R,header=','.join(self.quantities))
    #         np.savetxt(filename + '.bondorder',R)
