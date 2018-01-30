#
# nga.py
# contains essential classes for performing Neighborhood Graph Analysis
#
# Copyright (c) 2018 Wesley Reinhart.
# This file is part of the crayon project, released under the Modified BSD License.

from __future__ import print_function

import numpy as np
from emd import emd

import sys

sys.path.append('/home/wfr/freud/freud_0.6.3')
import freud

sys.path.append('/home/wfr/crayon/build')
import crayon

class Graph:
    R""" evaluates topology of neighborhood

    Args:
        A (array-like): adjacency matrix defining the neighborhood graph
    """
    def __init__(self,A):
        self.build(A)
    def build(self,A):
        # instantiate a Crayon::Graph object
        self.C = crayon.graph(A)
        self.adj = self.C.adj()
        # compute its Graphlet Degree Distribution
        self.gdd = self.C.gdd()
        # compute and normalize its Graphlet Degree Vector
        self.gdv = self.C.gdv()
        s = np.sum(self.gdv,1)
        s[s==0] = 1.
        self.ngdv = self.gdv / np.transpose( s * np.ones((self.gdv.shape[1],1)))
    def __sub__(self,other):
        R""" difference between this and another Graph, defined as the Earth Mover's Distance
        between normalized Graphlet Degree Vectors
        """
        return emd(self.ngdv,other.ngdv)
    def __eq__(self,other):
        R""" logical comparison between this and another Graph; checks if A - B == 0
        """
        return (self - other == 0.)
    def __str__(self):
        R""" hashable representation of the Graph, using the Graphlet Degree Distribution
        """
        return str(self.gdd.tolist())

class Snapshot:
    R""" identifies neighborhoods from simulation snapshot

    Args:
        xyz (array-like): N x 3 array of particle positions
        L (array-like): box dimensions in x, y, z
    """
    def __init__(self,xyz=None,L=None):
        # check that variable types and sizes match expected
        if xyz is not None:
            try:
                R = np.asarray(xyz)
            except:
                raise Exception('Error: xyz must be array-like')
            if R.shape[1] != 3:
                raise Exception('Error: xyz must have 3 columns')
        else:
            raise Exception('Error: must provide xyz (array of particle coordinates)')
        if L is not None:
            try:
                L = np.asarray(L)
            except:
                raise Exception('Error: L must be array-like')
            if len(L) != 3:
                raise Exception('Error: L must have 3 elements')
        else:
            raise Exception('Error: must provide L (box dimensions)')
        # nlAllowed = ['Delaunay','adaptiveCNA']
        # if nlType not in nlAllowed:
        #     raise Exception('Error: nlType must be one of the following: %s'%', '.join(nlAllowed))
        # for p in periodic:
        #     if p.lower() not in 'xyz':
        #         raise Exception('Error: periodic boundary conditions must be combination of x, y, and z')
        # assign snapshot data
        self.xyz = xyz
        self.L = L
        self.N = len(xyz)
        self.NL = None
        self.neighbors = None
    def buildNeighborList(self,rmax=40.,nnbr=16,strict_rcut=True):
        self.box = freud.box.Box(Lx=self.L[0],Ly=self.L[1],Lz=self.L[2],is2D=False)
        self.NL  = freud.locality.NearestNeighbors(rmax,nnbr,strict_cut=strict_rcut)
        self.NL.compute(self.box,self.xyz,self.xyz)
    def getNeighbors(self,nnbr=6,geoFactor=1.2071):
        if self.NL is None:
            self.buildNeighborList()
        Rsq = self.NL.getRsqList()
        Rsq[Rsq < 0] = np.nan
        self.Rsq = Rsq
        R6 = np.nanmean(Rsq[:,:nnbr],axis=1)
        self.Rcut = geoFactor**2. * R6
        NL = self.NL.getNeighborList()
        self.neighbors = []
        for i in range(self.N):
            self.neighbors.append(NL[i,self.Rsq[i,:]<self.Rcut[i]])
    def getNeighborhoods(self):
        if self.neighbors is None:
            self.getNeighbors()
        self.NN = []
        for i in range(self.N):
            n = len(self.neighbors[i])
            A = np.zeros((n,n),np.int8)
            idx = self.neighbors[i]
            for j in range(n):
                for k in range(j,n):
                    j_in_k = idx[j] in self.neighbors[idx[k]]
                    k_in_j = idx[k] in self.neighbors[idx[j]]
                    edge = int( j_in_k or k_in_j or j == k )
                    A[j,k] = edge
                    A[k,j] = edge
            self.NN.append(A)
        return self.NN

class Ensemble:
    def __init__(self):
        self.structures = {}
    def examine(self,S):
        NN = S.getNeighborhoods()
        # build graphs from neighborhoods
        for i in range(S.N):
            G = Graph(NN[i])
            self.encounter(G)
    def encounter(self,G):
        sig = str(G)
        new = True
        match = [s for s in self.structures.keys() if s == sig]
        # check graphs with matching GDD for true (GDV-EMD) equivalence
        for m in match:
            if self.structures[m] == G:
                new = False
                break
        if new:
            self.structures[sig] = G
    def coalesce(self,others):
        for other in others:
            for key in other:
                self.encounter(other[key])
