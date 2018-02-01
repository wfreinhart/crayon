#
# nga.py
# define core classes for Neighborhood Graph Analysis
#
# Copyright (c) 2018 Wesley Reinhart.
# This file is part of the crayon project, released under the Modified BSD License.

from __future__ import print_function
import sys
import inspect
import pickle
import zlib

from crayon import _crayon

import numpy as np
from emd import emd

from neighborlist import *

class Graph:
    R""" evaluates topology of neighborhood

    Args:
        A (array-like): adjacency matrix defining the neighborhood graph
    """
    def __init__(self,A):
        # instantiate a Crayon::Graph object
        self.C = _crayon.graph(A)
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
        R""" equality comparison between this and another Graph; checks if A - B == 0
        """
        return (self - other == 0.)
    def __ne__(self,other):
        R""" inequality comparison between this and another Graph; checks if A - B > 0
        """
        return (self - other > 0.)
    def __str__(self):
        R""" hashable representation of the Graph, using the Graphlet Degree Distribution
        """
        s_nodes = str(len(self.adj))
        s_edges = str(np.sum(self.adj))
        s_gdd = str(self.gdd.tolist())
        s = '%s:%s:%s'%(s_nodes,s_edges,s_gdd)
        return s
        # return zlib.compress(s)

class GraphLibrary:
    R""" handles sets of graphs from snapshots and ensembles of snapshots

    Args:
        (None)
    """
    def __init__(self,neighborhoods=None):
        self.graphs = {}
        self.counts = {}
        self.index = {}
        if neighborhoods is not None:
            self.build(neighborhoods)
    def build(self,neighborhoods):
        for i, nn in enumerate(neighborhoods):
            G = Graph(nn)
            self.encounter(G)
    def find(self,G):
        sig = str(G)
        try:
            return self.index[sig]
        except:
            return None
    def encounter(self,G,count=1,add=True):
        R""" adds a Graph object to the library and returns its index

        Args:
            sig (str): hashable signature of the graph
            g (Graph): Graph object to consider
            c (int): count to add to the library (i.e., number of observations from a Snapshot)

        Returns:
            idx (int): the index of this Graph (signature) in the library
        """
        sig = str(G)
        idx = None
        try:
            self.counts[sig] += count
        except:
            self.index[sig] = len(self.graphs)
            self.graphs[sig] = G
            self.counts[sig] = count
        if self.graphs[sig] != G:
            print(G.adj)
            print(self.graphs[sig].adj)
            print(self.graphs[sig] - G)
            raise RuntimeError('Found degenerate GDD: \n%s\n'%sig)
        return self.index[sig]
    def collect(self,others,counts=True):
        R""" merges other GraphLibrary objects into this one

        Args:
            others (list of GraphLibrary): GraphLibrary objects to merge into this one
        """
        if type(others) != list:
            others = list([others])
        if type(others[0]) != type(GraphLibrary()):
            raise TypeError('GraphLibrary.collect expects a list of GraphLibrary objects')
        # iterate over supplied library instances
        for other in others:
            for sig in other.graphs.keys():
                self.encounter(other.graphs[sig],count=(other.counts[sig] if counts else 0))

class Snapshot:
    R""" identifies neighborhoods from simulation snapshot

    Args:
        xyz (array-like): N x 3 array of particle positions
        L (array-like): box dimensions in x, y, z
        pbc (str): dimensions with periodic boundaries (defaults to 'xyz')
    """
    def __init__(self,xyz=None,L=None,pbc='xyz'):
        # check for valid particle positions
        if xyz is not None:
            try:
                R = np.asarray(xyz)
            except:
                raise TypeError('Error: xyz must be array-like')
            if R.shape[1] != 3:
                raise ValueError('Error: xyz must have 3 columns')
        else:
            raise RuntimeError('Error: must provide xyz (array of particle coordinates)')
        self.xyz = xyz
        self.N = len(xyz)
        # check for valid box size
        if L is not None:
            try:
                L = np.asarray(L)
            except:
                raise TypeError('Error: L must be array-like')
            if len(L) != 3:
                raise ValueError('Error: L must have 3 elements')
        else:
            raise RuntimeError('Error: must provide L (box dimensions)')
        self.L = L
        # check for valid periodic boundary conditions
        for p in pbc:
            if p.lower() not in 'xyz':
                raise ValueError('Error: periodic boundary conditions must be combination of x, y, and z')
        self.pbc = pbc
        # let NeighborList build neighbors
        self.neighbors = None

class Ensemble:
    def __init__(self):
        self.library = GraphLibrary()
        self.ldmap = dmap.DMap()
