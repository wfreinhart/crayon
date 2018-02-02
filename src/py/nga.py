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

from crayon import _crayon

import numpy as np
from emd import emd

try:
    import zlib
    sig_compression = True
except:
    sig_compression = False

sig_compression = False

from color import *
from dmap import *
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
        if sig_compression:
            return zlib.compress(s)
        else:
            return s

class GraphLibrary:
    R""" handles sets of graphs from snapshots and ensembles of snapshots

    Args:
        (None)
    """
    def __init__(self):
        self.graphs = {}
        self.counts = {}
        self.index = {}
        self.sigs = []
    def build(self,neighborhoods):
        nn_idx = []
        for i, nn in enumerate(neighborhoods):
            G = Graph(nn)
            nn_idx.append(self.encounter(G))
        nn_idx = np.asarray(nn_idx,dtype=np.int)
        nn_lookup = {}
        for i, sig in enumerate(self.graphs.keys()):
            if sig not in nn_lookup:
                nn_lookup[sig] = np.array([])
            nn_lookup[sig] = np.hstack((nn_lookup[sig],np.argwhere(nn_idx==self.index[sig]).flatten()))
        return nn_lookup
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
    def __init__(self,reader_input,reader=None,pbc='xyz',nl=None):
        # initialize class member variables
        self.neighbors = None
        self.adjacency = None
        self.library = None
        self.lookup = None
        # load from file
        if reader is None:
            self.load(reader_input)
            return None
        # read from generator function
        reader(self,reader_input)
        # check for valid periodic boundary conditions
        for p in pbc:
            if p.lower() not in 'xyz':
                raise ValueError('periodic boundary conditions must be combination of x, y, and z')
        self.pbc = pbc
        # check for valid NeighborList object
        if nl is not None:
            if type(nl) != type(NeighborList()):
                raise ValueError('nl must be a NeighborList object')
        else:
            raise RuntimeError('must provide a NeighborList object')
        self.nl = nl
    def buildNeighborhoods(self):
        self.neighbors = self.nl.getNeighbors(self)
    def buildAdjacency(self):
        self.adjacency = self.nl.getAdjacency(self)
    def buildLibrary(self):
        self.library = GraphLibrary()
        if self.adjacency is None:
            if self.neighbors is None:
                self.buildNeighborhoods()
            self.buildAdjacency()
        self.lookup = self.library.build(self.adjacency)
    def mapTo(self,library):
        for sig, idx in self.lookup.items():
            pass
    def save(self,filename,neighbors=False,adjacency=False,library=False):
        buff = {}
        if neighbors:
            buff['neighbors'] = self.neighbors
        if adjacency:
            buff['adjacency'] = self.adjacency
        if library:
            buff['library'] = self.library
            buff['lookup'] = self.lookup
        with open(filename,'wb') as fid:
            pickle.dump(buff,fid)
    def load(self,filename):
        with open(filename,'rb') as fid:
            buff = pickle.load(fid)
        if 'neighbors' in buff:
            self.neighbors = buff['neighbors']
        if 'adjacency' in buff:
            self.adjacency = buff['adjacency']
        if 'library' in buff:
            self.library = buff['library']
            self.lookup = buff['lookup']

class Ensemble:
    def __init__(self):
        self.library = GraphLibrary()
        self.dmap = DMap()
        self.lookups = {}
        self.dists = None
    def insert(self,idx,snap):
        if snap.library is None:
            snap.buildLibrary()
        self.library.collect(snap.library)
        self.lookups[idx] = snap.lookup
    def collect(self,others):
        if type(others) != list:
            others = list([others])
        if len(others) == 0:
            return
        if type(others[0]) != type(Ensemble()):
            raise TypeError('Ensemble.collect expects a list of Ensemble objects')
        # iterate over supplied library instances
        for other in others:
            self.library.collect(other.library)
            for key, val in other.lookups.items():
                if key in self.lookups:
                    print('Warning: duplicate lookup key detected during Ensemble.collect')
                self.lookups[key] = val
    def getColorMaps(self,cidx):
        c, c_map = compressColors(self.dmap.color_coords[:,cidx],delta=0.01)
        frames = self.lookups.keys()
        frames.sort()
        frame_maps = []
        for f in frames:
            N = np.sum(np.asarray([len(val) for key, val in self.lookups[f].items()]))
            frame_data = np.zeros(N)
            for key, val in self.lookups[f].items():
                frame_data[np.asarray(val,dtype=np.int)] = c_map[self.library.index[key]]
            frame_maps.append(frame_data)
        return c, frame_maps
