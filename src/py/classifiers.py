#
# classifiers.py
# define classifiers as basis for Neighborhood Graph Analysis
#
# Copyright (c) 2018 Wesley Reinhart.
# This file is part of the crayon project, released under the Modified BSD License.

from __future__ import print_function

from crayon import _crayon

import numpy as np

class Classifier:
    R""" abstract class defining a neighborhood topology
    """
    def __init__(self):
        self.s = None
    def __sub__(self,other):
        return None
    def __eq__(self,other):
        R""" equality comparison between this and another Classifier,
             simply checks if A - B == 0
        """
        return (self - other == 0.)
    def __ne__(self,other):
        R""" inequality comparison between this and another Classifier,
             simply checks if A - B > 0
        """
        return not self == other
    def __str__(self):
        R""" hashable representation of the Classifier, as specified
             by the constructor
        """
        return self.s

class Graph(Classifier):
    R""" evaluates topology of neighborhood as presented in a single graph

    Args:
        A (array): adjacency matrix defining the neighborhood graph
        k (int,optional): maximum graphlet size (default 5)
    """
    def __init__(self,A,k=5):
        if type(A) == tuple:
            self.sgdv = A[0]
            self.ngdv = A[1]
        else:
            self.build(A,k)
        # build a hashable representation of the graph
        self.s = str(self.sgdv.tolist()).replace(' ','')
    def build(self,A,k=5):
        R""" builds Graph from adjacency matrix and computes necessary quantities for NGA

        Args:
            A (array): adjacency matrix
            k (int,optional): maximum graphlet size (default 5)
        """
        # instantiate a Crayon::Graph object
        self.cpp = _crayon.neighborhood(A,k)
        # retrieve adjacency matrix
        self.adj = self.cpp.adj()
        # compute its Graphlet Degree Vector
        self.gdv = self.cpp.gdv()
        # convert node-wise to graph-wise graphlet frequencies
        self.sgdv = np.sum(self.gdv,axis=0)
        # weight GDV according to dependencies between orbits
        o = np.array([1, 2, 2, 2, 3, 4, 3, 3, 4, 3,
                      4, 4, 4, 4, 3, 4, 6, 5, 4, 5,
                      6, 6, 4, 4, 4, 5, 7, 4, 6, 6,
                      7, 4, 6, 6, 6, 5, 6, 7, 7, 5,
                      7, 6, 7, 6, 5, 5, 6, 8, 7, 6,
                      6, 8, 6, 9, 5, 6, 4, 6, 6, 7,
                      8, 6, 6, 8, 7, 6, 7, 7, 8, 5,
                      6, 6, 4],dtype=np.float)
        w = 1. - o / 73.
        self.ngdv = self.sgdv * w[:self.sgdv.shape[0]]
        self.ngdv = self.ngdv / max(float(np.sum(self.ngdv)),1.)
    def __sub__(self,other):
        R""" difference between this and another Graph, just the norm
        between graph-wide Graphlet Degree Vectors
        """
        return np.linalg.norm(self.ngdv-other.ngdv)

class Library:
    R""" handles sets of generic signatures from snapshots and ensembles of snapshots
    """
    def __init__(self):
        self.sigs   = []
        self.items = []
        self.counts = np.array([],dtype=np.int)
        self.sizes = np.array([],dtype=np.int)
        self.index = {}
        self.lookup = {}
    def build(self):
        return
    def find(self,item):
        R""" locate an object's signature in the Library

        Args:
            item (object): object to be located

        Returns:
            index (int): index of the object's signature
        """
        sig = str(item)
        try:
            return self.index[sig]
        except:
            return None
    def encounter(self,item,count=1,size=0,add=True):
        R""" adds an object to the library and returns its index

        Args:
            item (object): object to consider
            count (int,optional): count to add to the library (e.g., frequency from Snapshot) (default 1)
            add (bool,optional): should the item be added to the Library? (alternative is only find) (default True)

        Returns:
            idx (int): the index of the item's signature in the library
        """
        sig = str(item)
        try:
            idx = self.index[sig]
            self.counts[idx] += count
            self.sizes[idx] = max(self.sizes[idx],size)
        except:
            idx = len(self.items)
            self.sigs.append(sig)
            self.items.append(item)
            self.counts = np.append(self.counts,count)
            self.sizes = np.append(self.sizes,size)
            self.index[sig] = idx
        return idx
    def collect(self,others,counts=True,sizes=True):
        R""" merges other Library objects into this one

        Args:
            others (list of Library objects): Library objects to merge into this one
            counts (bool,optional): should the counts of the others be added together? (default True)
            sizes (bool,optional): should the maximum size of the others replace this one? (default True)
        """
        if type(others) != list:
            others = list([others])
        if type(others[0]) != type(self):
            raise TypeError('Library.collect expects a list of Library objects, but got %s != %s'%(str(type(others[0])),str(type(self))))
        # iterate over supplied library instances
        for other in others:
            for idx in range(len(other.items)):
                self.encounter(other.items[idx],
                               count=(other.counts[idx] if counts else 0),
                               size=(other.sizes[idx] if sizes else 0))

class GraphLibrary(Library):
    R""" handles sets of graphs from snapshots and ensembles of snapshots
    """
    def build(self,neighborhoods,k=5):
        R""" builds the GraphLibrary from neighborhoods

        Args:
            neighborhoods (list): list of neighborhoods to build from
            k (int,optional): maximum graphlet size (default 5)
        """
        g_idx = np.zeros(len(neighborhoods),dtype=np.int)
        for i, nn in enumerate(neighborhoods):
            G = Graph(nn,k)
            g_idx[i] = self.encounter(G)
        for i, sig in enumerate(self.sigs):
            if sig not in self.lookup:
                self.lookup[sig] = np.array([],dtype=np.int)
            self.lookup[sig] = np.hstack((self.lookup[sig],np.argwhere(g_idx==self.index[sig]).flatten()))
