#
# neighborlist.py
# build neighbor lists from particle positions
#
# Copyright (c) 2018 Wesley Reinhart.
# This file is part of the crayon project, released under the Modified BSD License.

from __future__ import print_function
import sys

import numpy as np

from crayon import _crayon

from scipy.cluster import hierarchy

try:
    import freud
    foundFreud = True
except:
    print('Warning: freud python module not found, neighborlist.AdaptiveCNA will not be available')
    foundFreud = False

class NeighborList:
    def __init__(self,second_shell=False):
        self.second_shell = second_shell
        self.setParams()
    def setParams(self):
        pass
    def getNeighbors(self,snap):
        return []
    # builds an adjacency matrix from the nearest neighbor list
    def particleAdjacency(i, NL):
        idx = NL[i].flatten()
        if self.second_shell:
            shell2 = []
            for j in range(len(idx)):
                shell2 += list(NL[idx[j]])
            idx = np.asarray(list(set(shell2)),dtype=np.int)
        n = len(idx)
        A = np.zeros((n,n),np.int8)
        for j in range(len(idx)):
            for k in range(len(idx)):
                A[j,k] = int( (idx[k] in NL[idx[j]].flatten()) or j == k )
        return A
    def getAdjacency(self,snap):
        adjacency = []
        for i in range(snap.N):
                adjacency.append(self.particleAdjacency(i,snap.neighbors))
        return adjacency

class AdaptiveCNA(NeighborList):
    def setParams(self,r_max=40.,max_nbr=16,strict_rcut=True,near_nbr=6,geo_factor=1.2071):
        self.r_max = r_max
        self.max_nbr = max_nbr
        self.strict_rcut = strict_rcut
        self.near_nbr = near_nbr
        self.geo_factor = geo_factor
        if not foundFreud:
            raise RuntimeError('neighborlist.AdaptiveCNA requires freud')
    def getNeighbors(self,snap):
        box = freud.box.Box(Lx=snap.L[0],Ly=snap.L[1],Lz=snap.L[2],is2D=False)
        nl  = freud.locality.NearestNeighbors(self.r_max,self.max_nbr,strict_cut=self.strict_rcut)
        nl.compute(box,snap.xyz,snap.xyz)
        Rsq = nl.getRsqList()
        Rsq[Rsq < 0] = np.nan
        R6 = np.nanmean(Rsq[:,:self.near_nbr],axis=1)
        Rcut = self.geo_factor**2. * R6
        nl = nl.getNeighborList()
        neighbors = []
        for i in range(snap.N):
            neighbors.append(nl[i,Rsq[i,:]<Rcut[i]])
        return neighbors

class Voronoi(NeighborList):
    def setParams(self,clustering=True,cluster_method='centroid',cluster_ratio=0.25):
        self.clustering = clustering
        self.cluster_method = cluster_method
        self.cluster_ratio = cluster_ratio
    # filter neighbors by hierarchical clustering
    def filterNeighbors(self,idx,neighbors,snap):
        # get neighbors from triangulation
        nn = np.asarray(neighbors[idx],dtype=np.int)
        # get displacement vectors
        d_vec = snap.xyz[nn,:] - snap.xyz[idx,:]
        # wrap them according to boundary conditions specified in Snapshot
        pbc = np.asarray([dim in snap.pbc for dim in 'xyz'],dtype=np.float)
        d_vec -= snap.L * np.round( d_vec / snap.L * pbc)
        # sort neighbors by increasing distance
        d_nbr = np.sqrt(np.sum((d_vec)**2.,1))
        order = np.argsort(d_nbr)
        nn = np.array(nn)[order]
        d_nbr = d_nbr[order]
        X = d_nbr.reshape(-1,1)
        # exclude far-away particles by clustering
        Z = hierarchy.linkage(X,self.cluster_method)
        c = hierarchy.fcluster(Z,self.cluster_ratio*d_nbr[0],criterion='distance')
        h_base = np.argwhere(c == c[0]).flatten()
        nn = np.hstack(([idx],nn[h_base]))
        return nn
    # compute Delaunay triangulation with Voro++ library
    def getNeighbors(self,snap):
        nl = _crayon.voropp(snap.xyz, snap.L, 'x' in snap.pbc, 'y' in snap.pbc, 'z' in snap.pbc)
        neighbors = []
        for idx in range(snap.N):
            if self.clustering:
                nn = self.filterNeighbors(idx,nl,snap)
            else:
                nn = nl[idx]
            neighbors.append(np.array(nn,dtype=np.int))
        return neighbors
