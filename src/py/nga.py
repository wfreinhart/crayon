#
# nga.py
# define core classes for Neighborhood Graph Analysis
#
# Copyright (c) 2018 Wesley Reinhart.
# This file is part of the crayon project, released under the Modified BSD License.

from __future__ import print_function

from crayon import _crayon
from crayon import parallel
from crayon import neighborlist
from crayon import color
from crayon import io
from crayon import dmap

import numpy as np
from emd import emd

try:
    import pickle
    allow_binary = True
except:
    allow_binary = False

try:
    import zlib
    sig_compression = True
except:
    sig_compression = False

sig_compression = False

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
        s = np.sum(self.gdv,1)
        s[s==0] = 1.
        self.ngdv = self.gdv / np.transpose( s * np.ones((self.gdv.shape[1],1)))
    def __sub__(self,other):
        R""" difference between this and another Graph, defined as the Earth Mover's Distance
        between Graphlet Degree Vectors
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
        self.sigs   = []
        self.graphs = []
        self.counts = np.array([])
        self.index = {}
    def build(self,neighborhoods):
        nn_idx = np.zeros(len(neighborhoods),dtype=np.int)
        for i, nn in enumerate(neighborhoods):
            G = Graph(nn)
            nn_idx[i] = self.encounter(G)
        nn_lookup = {}
        for i, sig in enumerate(self.sigs):
            if sig not in nn_lookup:
                nn_lookup[sig] = np.array([],dtype=np.int)
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
        try:
            idx = self.index[sig]
            self.counts[idx] += count
        except:
            idx = len(self.graphs)
            self.sigs.append(sig)
            self.graphs.append(G)
            self.counts = np.append(self.counts,count)
            self.index[sig] = idx
        if self.graphs[idx] != G:
            print(G.adj)
            print(self.graphs[idx].adj)
            print(self.graphs[idx] - G)
            raise RuntimeError('Found degenerate GDD: \n%s\n'%sig)
        return idx
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
            for idx in range(len(other.graphs)):
                self.encounter(other.graphs[idx],count=(other.counts[idx] if counts else 0))

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
            filename = reader_input
            try:
                self.load(reader_input)
                return None
            except:
                if '.xml' in filename:
                    reader = io.readXML
                elif '.gsd' in filename:
                    reader = io.readGSD
                elif '.xyz' in filename:
                    reader = io.readXYZ
        # read from generator function
        reader(self,reader_input)
        # check for valid periodic boundary conditions
        for p in pbc:
            if p.lower() not in 'xyz':
                raise ValueError('periodic boundary conditions must be combination of x, y, and z')
        self.pbc = pbc
        # check for valid NeighborList object
        if nl is not None:
            if type(nl) != type(neighborlist.NeighborList()):
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
        if type(library) != type(GraphLibrary()):
            raise ValueError('must supply a GraphLibrary object')
        m = np.zeros(self.N,dtype=np.int) * np.nan
        for sig, idx in self.lookup.items():
            if sig in library.index:
                m[idx] = library.index[sig]
        return m
    def wrap(self,v):
        pbc = np.asarray([dim in self.pbc for dim in 'xyz'],dtype=np.float)
        w = v - self.L * np.round( v / self.L * pbc)
        return w
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
            self.N = len(self.neighbors)
        if 'adjacency' in buff:
            self.adjacency = buff['adjacency']
            self.N = len(self.adjacency)
        if 'library' in buff:
            self.library = buff['library']
            self.lookup = buff['lookup']

class Ensemble:
    def __init__(self):
        self.library = GraphLibrary()
        self.dmap = None
        self.lookups = {}
        self.sigs = []
        self.graphs = []
        self.dists = None
        self.comm, self.size, self.rank, self.master = parallel.info()
        self.p = parallel.ParallelTask()
    def neighborhoodsFromFile(self,filenames,nl):
        self.filenames = filenames
        local_file_idx  = parallel.partition(range(len(self.filenames)))
        for f in local_file_idx:
            filename = self.filenames[f]
            print('rank %d of %d will process %s'%(self.rank,self.size,filename))
            # create snapshot instance and build neighborhoods
            snap = Snapshot(filename,pbc='xyz',nl=nl)
            self.insert(f,snap)
            snap.save(filename + '.nga',adjacency=True,neighbors=True)
        print('rank %d tasks complete, found %d unique graphs'%(self.rank,len(self.library.graphs)))
        self.collect()
    def insert(self,idx,snap):
        if snap.library is None:
            snap.buildLibrary()
        self.library.collect(snap.library)
        self.lookups[idx] = snap.lookup
    def collect(self):
        others = self.p.gatherData(self)
        if not self.master:
            return
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
        print('collection complete, found %d unique graphs'%(len(self.library.graphs)))
    def prune(self,min_freq=None):
        if not self.master:
            return
        try:
            min_freq = int(min_freq)
        except:
            raise RuntimeError('Must specify min_freq, and it must be castable to int')
        n = len(self.library.sigs)
        self.lm_idx = np.argwhere(self.library.counts >= min_freq).flatten()
        m = len(self.lm_idx)
        self.lm_sigs = [self.library.sigs[idx] for idx in self.lm_idx]
        print('using %d archetypal graphs as landmarks for %d less common ones'%(m,n-m))
    def getColorMaps(self,cidx):
        c, c_map = color.compressColors(self.dmap.color_coords[:,cidx],delta=0.001)
        frames = self.lookups.keys()
        frames.sort()
        frame_maps = []
        for f in frames:
            N = np.sum(np.asarray([len(val) for key, val in self.lookups[f].items()]))
            frame_data = np.zeros(N,dtype=np.int)
            for key, val in self.lookups[f].items():
                frame_data[np.asarray(val,dtype=np.int)] = self.library.index[key]
            frame_maps.append(frame_data)
        return c, c_map, frame_maps
    def computeDists(self):
        # use a master-slave paradigm for load balancing
        task_list = []
        if self.master:
            n = len(self.library.sigs)
            m = len(self.lm_sigs)
            self.dists = np.zeros( (n,m) ) + np.Inf # designate null values with Inf
            for i in range(n):
                for j in range(m):
                    task_list.append( (i,self.lm_idx[j]) )
        # perform graph matching in parallel using MPI
        graphs = self.p.shareData(self.library.graphs)
        eval_func = lambda task, data: data[task[0]] - data[task[1]]
        result_list = self.p.computeQueue(function=eval_func,
                                          tasks=task_list,
                                          reports=10)
        if self.master:
            # convert results into numpy array
            for k in range(len(result_list)):
                i, j = task_list[k]
                jid = np.argwhere(self.lm_idx == j)[0]
                d = result_list[k]
                self.dists[i,jid] = d
            self.dists = self.dists / np.max(self.dists)
    def autoColor(self,prefix='draw_colors',sigma=1.0,VMD=False,Ovito=False,similarity=True):
        coms = None
        if self.master:
            coms, best = self.dmap.uncorrelatedTriplets()
            print('probable best eigenvector triplet is %s'%str(coms[best]))
        coms = self.p.shareData(coms)
        self.colorTriplets(coms,prefix=prefix,sigma=sigma,VMD=VMD,Ovito=Ovito,similarity=similarity)
    def colorTriplets(self,trips,prefix='draw_colors',sigma=1.0,VMD=False,Ovito=False,similarity=True):
        # share data among workers
        colors = []
        color_maps = []
        frame_maps = []
        color_coords = None
        if self.master:
            color_coords = self.dmap.color_coords
            for trip in trips:
                c, cm, fm = self.getColorMaps(np.array(trip))
                colors.append(c)
                color_maps.append(cm)
                frame_maps.append(fm)
        colors = self.p.shareData(colors)
        frame_maps = self.p.shareData(frame_maps)
        color_maps = self.p.shareData(color_maps)
        color_coords = self.p.shareData(color_coords)
        # compute cluster similarity
        local_file_idx  = parallel.partition(range(len(self.filenames)))
        for f in local_file_idx:
            filename = self.filenames[f]
            snap = Snapshot(filename + '.nga')
            for t, trip in enumerate(trips):
                if similarity:
                    sim = color.neighborSimilarity(frame_maps[t][f],snap.neighbors,color_coords[:,np.array(trip)])
                else:
                    sim = color_coords[frame_maps[t][f].reshape(-1,1),np.array(trip)]
                mapped_color = color_maps[t][frame_maps[t][f]].reshape(-1,1)
                f_dat = np.hstack((mapped_color,sim))
                np.savetxt(filename + '_%d%d%d.cmap'%trip, f_dat)
        if not self.master:
            return
        # write visualization scripts
        for t, trip in enumerate(trips):
            if VMD:
                color.writeVMD('%s_%d%d%d.tcl'%(prefix,trip[0],trip[1],trip[2]),
                               self.filenames, colors[t], trip, f_dat.shape[1], sigma=sigma,
                               swap=('/home/wfr/','/Users/wfr/mountpoint/'))
    def buildDMap(self):
        if self.master:
            self.dmap = dmap.DMap()
            self.dmap.set_params()
            self.dmap.build(self.dists,landmarks=self.lm_idx)
            print('Diffusion map construction complete')
            self.dmap.write()
