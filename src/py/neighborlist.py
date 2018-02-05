#
# neighborlist.py
# build neighbor lists from particle positions
#
# Copyright (c) 2018 Wesley Reinhart.
# This file is part of the crayon project, released under the Modified BSD License.

from __future__ import print_function
import sys

import numpy as np

from scipy.spatial import Delaunay as triangulate
from scipy.spatial.distance import pdist
from scipy.cluster import hierarchy

try:
    import freud
    foundFreud = True
except:
    print('Warning: freud python module not found, neighborlist.AdaptiveCNA will not be available')
    foundFreud = False

# builds an adjacency matrix from the nearest neighbor list
def neighborsToAdjacency(i, NL):
    n = len(NL[i])
    A = np.zeros((n,n),np.int8)
    idx = NL[i].flatten()
    for j in range(len(idx)):
        for k in range(len(idx)):
            A[j,k] = int( (idx[k] in NL[idx[j]].flatten()) or j == k )
    return A

class NeighborList:
    def __init__(self):
        self.setParams()
    def setParams(self):
        pass
    def getNeighbors(self,snap):
        return []
    def getAdjacency(self,snap):
        adjacency = []
        for i in range(snap.N):
            adjacency.append(neighborsToAdjacency(i,snap.neighbors))
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
    def setParams(self,cluster_method='centroid',cluster_ratio=0.25):
        self.cluster_method = cluster_method
        self.cluster_ratio = cluster_ratio
    # find set of neighbors from triangulation mesh
    def triNeighbors(self,idx,tri):
        try:
            verts = tri.vertices
        except:
            verts = tri.simplices
        tri_conn = verts[np.sum(verts==idx,1)>0,:].flatten()
        nn = list(set(tri_conn[tri_conn != idx]))
        return nn
    # filter neighbors by hierarchical clustering
    def filterNeighbors(self,idx,W,tri):
        # get neighbors from triangulation
        nn = np.array( self.triNeighbors(idx,tri) )
        # sort neighbors by increasing distance
        d_nbr = np.sqrt(np.sum((W[nn,:] - W[idx,:])**2.,1))
        order = np.argsort(d_nbr)
        nn = np.array(nn)[order]
        d_nbr = d_nbr[order]
        X = d_nbr.reshape(-1,1)
        # exclude far-away particles by clustering
        Z = hierarchy.linkage(X,self.cluster_method)
        c = hierarchy.fcluster(Z,self.cluster_ratio*d_nbr[0],criterion='distance')
        h_base = np.argwhere(c == c[0]).flatten()
        nn = nn[h_base]
        return nn
    # return positions of particles and their nearest images
    def boxImage(self,snap):
        M = np.hstack((0.,snap.L))
        d_max = np.sqrt(np.sum(M**2.))
        I = np.reshape(np.arange(snap.xyz.shape[0]),(-1,1))
        W = np.hstack((I,snap.xyz))
        dim_keys = {'x': 0, 'y': 1, 'z': 2}
        pdim = [dim_keys[x] for x in snap.pbc]
        for dim in pdim:
            M_dim = np.zeros(4)
            M_dim[dim+1] = M[dim+1]
            neg_half = W[:,dim+1] < 0
            pos_half = W[:,dim+1] > 0
            neg_img  = W[neg_half,:] + M_dim
            pos_img  = W[pos_half,:] - M_dim
            W = np.vstack((W,neg_img,pos_img))
        return W[:,0], W[:,1:]
    # compute Delaunay triangulation
    def getNeighbors(self,snap):
        I, W = self.boxImage(snap)
        tri = triangulate(W)
        neighbors = []
        for idx in range(snap.N):
            nn = I[np.asarray( self.filterNeighbors(idx,W,tri) )].flatten()
            nn_unique = np.array(list(set(nn)),dtype=np.int)
            neighbors.append(nn_unique)
        return neighbors
