#
# classifiers.py
# define classifiers as basis for Neighborhood Graph Analysis
#
# Copyright (c) 2018 Wesley Reinhart.
# This file is part of the crayon project, released under the Modified BSD License.

from __future__ import print_function

from crayon import _crayon

import numpy as np

from emd import emd

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
        # compute its Graphlet Degree Vector
        self.gdv = self.C.gdv()
        # convert node-wise to graph-wise graphlet frequencies
        self.ngdv = np.sum(self.gdv,0) / max(float(np.sum(self.gdv)),1.)
        # build a hashable representation of the graph
        s_nodes = str(len(self.adj))
        s_edges = str(np.sum(self.adj))
        s_gdv = str(np.sum(self.gdv,0).tolist()).replace(' ','')
        self.s = '%s:%s:%s'%(s_nodes,s_edges,s_gdv)
    def __sub__(self,other):
        R""" difference between this and another Graph, just the norm
        between graph-wide Graphlet Degree Vectors
        """
        return np.linalg.norm(self.ngdv-other.ngdv)
    def __eq__(self,other):
        R""" equality comparison between this and another Graph; checks if A - B == 0
        """
        return (self - other == 0.)
    def __ne__(self,other):
        R""" inequality comparison between this and another Graph; checks if A - B > 0
        """
        return (self - other > 0.)
    def __str__(self):
        R""" hashable representation of the Graph, using the Graphlet Degree Vector
        """
        return self.s

class Pattern:
    R""" evaluates sets of particle topologies encountered in a neighborhood

    Args:
        S (str): string containing the Graphlet Degree Vectors from all Graphs in the Pattern
    """
    def __init__(self,S):
        self.s = S
        ngdv = []
        for s in S.split('/'):
            s_gdv = s.split(':')[-1]
            gdv = np.array(s_gdv.replace('[','').replace(']','').split(','),dtype=np.int)
            ngdv.append( gdv / max(float(np.sum(gdv)),1.) )
        self.ngdv = np.array(ngdv)
    def __sub__(self,other):
        R""" difference between this and another Pattern, using Earth Movers Distance
        on the set of normalized GDVs
        """
        ni = len(self.ngdv)
        nj = len(other.ngdv)
        D = np.zeros((ni,nj))
        for i, igdv in enumerate(self.ngdv):
            for j, jgdv in enumerate(other.ngdv):
                D[i,j] = np.linalg.norm(igdv-jgdv)
        return emd(range(ni),range(nj),distance='precomputed',D=D)
    def __eq__(self,other):
        R""" equality comparison between this and another Pattern; checks if A - B == 0
        """
        return (self - other == 0.)
    def __ne__(self,other):
        R""" inequality comparison between this and another Patter; checks if A - B > 0
        """
        return (self - other > 0.)
    def __str__(self):
        R""" hashable representation of the Pattern, using the set of GDVs
        """
        return self.s

class Library:
    R""" handles sets of generic signatures from snapshots and ensembles of snapshots

    Args:
        (None)
    """
    def __init__(self):
        self.sigs   = []
        self.items = []
        self.counts = np.array([])
        self.index = {}
        self.lookup = {}
    def build(self):
        return
    def find(self,item):
        sig = str(item)
        try:
            return self.index[sig]
        except:
            return None
    def encounter(self,item,count=1,add=True):
        R""" adds an object to the library and returns its index

        Args:
            item (generic): object to consider
            count (int) (optional): count to add to the library (i.e., number of observations from a Snapshot) (default 1)
            add (bool) (optional): should the item be added to the library? (default True)

        Returns:
            idx (int): the index of this item (signature) in the library
        """
        sig = str(item)
        try:
            idx = self.index[sig]
            self.counts[idx] += count
        except:
            idx = len(self.items)
            self.sigs.append(sig)
            self.items.append(item)
            self.counts = np.append(self.counts,count)
            self.index[sig] = idx
        return idx
    def collect(self,others,counts=True):
        R""" merges other Library objects into this one

        Args:
            others (list of Library objects): Library objects to merge into this one
        """
        if type(others) != list:
            others = list([others])
        if type(others[0]) != type(self):
            raise TypeError('Library.collect expects a list of Library objects, but got %s != %s'%(str(type(others[0])),str(type(self))))
        # iterate over supplied library instances
        for other in others:
            for idx in range(len(other.items)):
                self.encounter(other.items[idx],
                               count=(other.counts[idx] if counts else 0))

class GraphLibrary(Library):
    R""" handles sets of graphs from snapshots and ensembles of snapshots

    Args:
        (None)
    """
    def build(self,neighborhoods):
        g_idx = np.zeros(len(neighborhoods),dtype=np.int)
        for i, nn in enumerate(neighborhoods):
            G = Graph(nn)
            g_idx[i] = self.encounter(G)
        for i, sig in enumerate(self.sigs):
            if sig not in self.lookup:
                self.lookup[sig] = np.array([],dtype=np.int)
            self.lookup[sig] = np.hstack((self.lookup[sig],np.argwhere(g_idx==self.index[sig]).flatten()))

class PatternLibrary(Library):
    R""" handles sets of patterns from snapshots and ensembles of snapshots

    Args:
        (None)
    """
    def build(self,neighbors,graph_sigs,graph_map):
        p_idx = np.zeros(len(neighbors),dtype=np.int)
        for i in range(len(neighbors)):
            participants = [int(x) for x in np.unique(graph_map[neighbors[i]])]
            S = ' / '.join(['%s'%graph_sigs[x] for x in participants])
            P = Pattern(S)
            p_idx[i] = self.encounter(P)
        print('Found %d unique patterns'%len(self.sigs))
        for i, sig in enumerate(self.sigs):
            if sig not in self.lookup:
                self.lookup[sig] = np.array([],dtype=np.int)
            self.lookup[sig] = np.hstack((self.lookup[sig],np.argwhere(p_idx==self.index[sig]).flatten()))
