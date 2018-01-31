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
    def __init__(self):
        self.graphs = {}
        self.counts = {}
        self.index = {}
    def encounter(self,G,count=1):
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
    def collect(self,others):
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
                self.encounter(other.graphs[sig],count=other.counts[sig])

class Snapshot:
    R""" identifies neighborhoods from simulation snapshot

    Args:
        xyz (array-like): N x 3 array of particle positions
        L (array-like): box dimensions in x, y, z
        pbc (str): dimensions with periodic boundaries (defaults to 'xyz')
    """
    def __init__(self,xyz=None,L=None,pbc='xyz',nl_type='adaptiveCNA'):
        # check that variable types and sizes match expected
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
        self.pbc = pbc
        nlAllowed = ['delaunay','adaptivecna']
        if nl_type.lower() not in nlAllowed:
            raise ValueError('Error: nlType must be one of the following: %s'%', '.join(nlAllowed))
        for p in pbc:
            if p.lower() not in 'xyz':
                raise ValueError('Error: periodic boundary conditions must be combination of x, y, and z')
        # assign snapshot data
        self.NL = AdaptiveCNA(self)
        # self.NL = Voronoi(self)
        self.NN = None
        self.library = GraphLibrary()
        self.graph_index = np.zeros(self.N) - 1
    def getNeighborhoods(self):
        if self.NL.neighbors is None:
            self.NL.build()
        self.NN = []
        for i in range(self.N):
            self.NN.append(neighborsToAdjacency(i,self.NL.neighbors))
        return self.NN
    def buildLibrary(self):
        R""" constructs neighborhood Graphs from nearest neighbors of all particles and adds them
             to a Snapshot-specific GraphLibrary
        """
        if self.NN is None:
            self.getNeighborhoods()
        # build graphs from neighborhoods
        for i in range(self.N):
            G = Graph(self.NN[i])
            self.graph_index[i] = self.library.encounter(G)
    def save(self,filename,graphs=True,neighborhoods=True):
        with open(filename,'wb') as fid:
            buff = {}
            if graphs:
                buff['graph_adj'] = [g.adj for key, g in self.library.graphs.items()]
                buff['graph_index'] = self.graph_index
            if neighborhoods:
                buff['NN'] = self.NN
            pickle.dump(buff,fid)
    def load(self,filename):
        with open(filename,'rb') as fid:
            buff = pickle.load(fid)
            if 'graph_adj' in buff and 'graph_index' in buff:
                self.graph_index = buff['graph_index']
                for i, adj in enumerate(buff['graph_adj']):
                    G = Graph(adj)
                    sig = str(G)
                    self.library.index[sig] = len(self.library.graphs)
                    self.library.graphs[sig] = G
                    self.library.counts[sig] = np.sum(self.graph_index == i)
            if 'NN' in buff:
                self.NN = buff['NN']

class Ensemble:
    def __init__(self):
        self.library = GraphLibrary()
        self.ldmap = dmap.DMap()
