#
# nga.py
# contains essential classes for performing Neighborhood Graph Analysis
#
# Copyright (c) 2018 Wesley Reinhart.
# This file is part of the crayon project, released under the Modified BSD License.

from __future__ import print_function

import numpy as np
from emd import emd

import pickle

import sys

sys.path.append('/home/wfr/crayon/build')
import crayon

from util_neighborlist import *

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

class GraphLibrary:
    def __init__(self):
        self.library = {}
        self.index_map = {}
    def encounter(self,key,entry):
        sig = key
        G = entry['graph']
        C = entry['count']
        sigs = np.asarray(self.library.keys(),dtype=np.str)
        match = np.argwhere( sigs == sig ).flatten()
        # match = [s for s in self.library.keys() if s == sig]
        # check graphs with matching GDD for true (GDV-EMD) equivalence
        idx = None
        for m in match:
            if self.library[sigs[m]]['graph'] == G:
                idx = m
                self.library[sigs[m]]['count'] += C
                break
        if idx is None:
            self.library[sig] = {'graph': G, 'count': C}
            idx = len(sigs)
            self.index_map[idx] = sig
        return idx
    def collect(self,others):
        if type(others) != list:
            others = list([others])
        # iterate over supplied library instances
        for other in others:
            for key in other:
                self.encounter(key,other[key])

class Snapshot(GraphLibrary):
    R""" identifies neighborhoods from simulation snapshot

    Args:
        xyz (array-like): N x 3 array of particle positions
        L (array-like): box dimensions in x, y, z
        pbc (str): dimensions with periodic boundaries (defaults to 'xyz')
    """
    def __init__(self,xyz=None,L=None,pbc='xyz'):
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
        self.xyz = xyz
        self.N = len(xyz)
        if L is not None:
            try:
                L = np.asarray(L)
            except:
                raise Exception('Error: L must be array-like')
            if len(L) != 3:
                raise Exception('Error: L must have 3 elements')
        else:
            raise Exception('Error: must provide L (box dimensions)')
        self.L = L
        self.pbc = pbc
        # nlAllowed = ['Delaunay','adaptiveCNA']
        # if nlType not in nlAllowed:
        #     raise Exception('Error: nlType must be one of the following: %s'%', '.join(nlAllowed))
        # for p in periodic:
        #     if p.lower() not in 'xyz':
        #         raise Exception('Error: periodic boundary conditions must be combination of x, y, and z')
        # assign snapshot data
        self.NL = AdaptiveCNA(self)
        # self.NL = Voronoi(self)
        self.NN = None
        self.library = {}
        self.index_map = {}
        self.graph_index = np.zeros(self.N) - 1
    def getNeighborhoods(self):
        if self.NL.neighbors is None:
            self.NL.build()
        self.NN = []
        for i in range(self.N):
            self.NN.append(neighborsToAdjacency(i,self.NL.neighbors))
        return self.NN
    def buildLibrary(self):
        if self.NN is None:
            self.getNeighborhoods()
        self.graphs = {}
        self.counts = {}
        self.graph_index = np.zeros(self.N)
        # build graphs from neighborhoods
        for i in range(self.N):
            G = Graph(self.NN[i])
            key = str(G)
            entry = {'graph': G, 'count': 1}
            self.graph_index[i] = self.encounter(key,entry)

class Ensemble(GraphLibrary):
    def __init__(self):
        self.library = {}
        self.index_map = {}
