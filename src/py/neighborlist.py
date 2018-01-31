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
    def __init__(self,snap):
        self.set_params()
        self.snap = snap
        self.neighbors = None
    def set_params(self):
        pass
    def build(self):
        self.neighbors = []

class AdaptiveCNA(NeighborList):
    def set_params(self,r_max=40.,max_nbr=16,strict_rcut=True,near_nbr=6,geo_factor=1.2071):
        self.r_max = r_max
        self.max_nbr = max_nbr
        self.strict_rcut = strict_rcut
        self.near_nbr = near_nbr
        self.geo_factor = geo_factor
        if not foundFreud:
            raise RuntimeError('neighborlist.AdaptiveCNA requires freud')
    def build(self):
        self.f_box = freud.box.Box(Lx=self.snap.L[0],Ly=self.snap.L[1],Lz=self.snap.L[2],is2D=False)
        self.f_nl  = freud.locality.NearestNeighbors(self.r_max,self.max_nbr,strict_cut=self.strict_rcut)
        self.f_nl.compute(self.f_box,self.snap.xyz,self.snap.xyz)
        Rsq = self.f_nl.getRsqList()
        Rsq[Rsq < 0] = np.nan
        R6 = np.nanmean(Rsq[:,:self.near_nbr],axis=1)
        Rcut = self.geo_factor**2. * R6
        nl = self.f_nl.getNeighborList()
        self.neighbors = []
        for i in range(self.snap.N):
            self.neighbors.append(nl[i,Rsq[i,:]<Rcut[i]])

class Voronoi(NeighborList):
    def set_params(self,cluster_method='centroid',cluster_ratio=0.25):
        self.cluster_method = cluster_method
        self.cluster_ratio = cluster_ratio
    # find set of neighbors from triangulation mesh
    def tri_neighbors(self,idx):
        try:
            verts = self.tri.vertices
        except:
            verts = self.tri.simplices
        tri_conn = verts[np.sum(verts==idx,1)>0,:].flatten()
        nn = list(set(tri_conn[tri_conn != idx]))
        return nn
    # filter neighbors by hierarchical clustering
    def filter_neighbors(self,idx):
        # get neighbors from triangulation
        nn = np.array( self.tri_neighbors(idx) )
        # sort neighbors by increasing distance
        d_nbr = np.sqrt(np.sum((self.W[nn,:] - self.W[idx,:])**2.,1))
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
    def box_image(self):
        M = np.hstack((0.,self.snap.L))
        d_max = np.sqrt(np.sum(M**2.))
        I = np.reshape(np.arange(self.snap.xyz.shape[0]),(-1,1))
        W = np.hstack((I,self.snap.xyz))
        dim_keys = {'x': 0, 'y': 1, 'z': 2}
        pdim = [dim_keys[x] for x in self.snap.pbc]
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
    def build(self):
        self.I, self.W = self.box_image()
        self.tri = triangulate(self.W)
        self.neighbors = []
        for idx in range(self.snap.N):
            nn = self.I[np.asarray( self.filter_neighbors(idx) )].flatten()
            nn_unique = np.array(list(set(nn)),dtype=np.int)
            self.neighbors.append(nn_unique)
