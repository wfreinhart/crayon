#
# nga.py
# define classes implementing Neighborhood Graph Analysis
#     for molecular simulation snapshots
#
# Copyright (c) 2018 Wesley Reinhart.
# This file is part of the crayon project, released under the Modified BSD License.

from __future__ import print_function

from crayon import _crayon
from crayon import classifiers
from crayon import parallel
from crayon import neighborlist
from crayon import dmap
from crayon import io
from crayon import util

import numpy as np
from scipy.cluster import hierarchy
import pickle

class Snapshot:
    R""" holds necessary data from a simulation snapshot and handles
         neighborlist generation and graph library construction

    Args:
        xyz (array): `Nx3` array containing `xyz` coordinates of particles
        box (array): box dimensions (`Lx,Ly,Lz`)
        nl (NeighborList): reference to a class capable of building the neighbor list
        pbc (str) (optional): dimensions with periodic boundaries (defaults to 'xyz')
    """
    def __init__(self,xyz=None,box=None,nl=None,pbc='xyz'):
        # initialize class member variables
        self.neighbors = None
        self.adjacency = None
        self.library = None
        # default options for building libraries
        self.global_mode = False
        self.graphlet_k = 5
        self.cluster = True
        self.cluster_thresh = None
        self.n_shells = 1
        # assign attributes
        self.xyz = xyz
        self.box = box
        if self.xyz is not None:
            self.N = len(self.xyz)
        else:
            self.N = None
        # check for valid NeighborList object
        if nl is not None:
            if type(nl) != type(neighborlist.NeighborList()):
                raise ValueError('nl must be a NeighborList object')
            self.nl = nl
        # check for valid periodic boundary conditions
        for p in pbc:
            if p.lower() not in 'xyz':
                raise ValueError('periodic boundary conditions must be combination of x, y, and z')
        self.pbc = pbc
        # auto-detect 2D configuration
        if self.xyz is not None:
            span = np.max(self.xyz,axis=0) - np.min(self.xyz,axis=0)
            dims = 'xyz'
            for i, s in enumerate(span):
                if s < 1e-4:
                    print('detected 2D configuration')
                    # force values for better compatibility with Voro++
                    self.box[i] = 1.
                    self.xyz[:,i] = 0.
                    self.pbc = self.pbc.replace(dims[i],'')
    def buildNeighborhoods(self):
        R""" requests neighborhoods from the supplied NeighborList class
        """
        self.neighbors = self.nl.getNeighbors(self)
    def buildAdjacency(self):
        R""" build adjacency matrices from pre-computed neighbors
        """
        if self.neighbors is None:
            self.buildNeighborhoods()
        if self.global_mode:
            self.adjacency = neighborlist.Network(self,k=self.graphlet_k)
        else:
            self.adjacency = _crayon.buildGraphs(self.neighbors,self.n_shells)
    def parseOptions(self,options):
        R"""

        Args:
            options (dictionary): kwargs from buildLibrary()
        """
        # use a global Network instead of local Neighborhoods?
        if 'global_mode' in options:
            self.global_mode = options['global_mode']
        # choose size of graphlets
        if 'graphlet_k' in options:
            self.graphlet_k = options['graphlet_k']
        # perform clustering to find relevant structures?
        if 'cluster' in options:
            self.cluster = options['cluster']
        # threshold radius in dmap space to allow in same cluster
        if 'cluster_thresh' in options:
            self.cluster_thresh = options['cluster_thresh']
        # number of neighbor shells to use (forced)
        if 'n_shells' in options:
            self.n_shells = options['n_shells']
    def buildLibrary(self,**kwargs):
        R""" build a GraphLibrary from all Graphs observed in this Snapshot

        Args:
            global_mode (bool,optional): use the graphlet decomposition from the entire graph? (default False)
            n_shells (int, optional): number of shells to include in the neighborhoods (default 1)
            graphlet_k (int,optional): maximum graphlet size (default 5)
            cluster (bool,optional): use hierarchical clustering in the NeighborList? (default True)
            cluster_thresh (float,optional): fractional distance between clusters in hierarchical clustering (default 0.25)

        """
        self.parseOptions(kwargs)
        self.library = classifiers.GraphLibrary()
        if self.adjacency is None:
            if self.neighbors is None:
                self.buildNeighborhoods()
            self.buildAdjacency()
        self.library.build(self.adjacency,k=self.graphlet_k)
        if self.cluster:
            self.library.sizes = neighborlist.largest_clusters(self,self.library,self.cluster_thresh)
    def mapTo(self,library):
        R""" obtain a mapping between the Snapshot's GraphLibrary and another GraphLibrary

        Args:
            library (Library): the Library to map particles onto

        Returns:
            m (array): the index of each particle in the provided Library
        """
        m = np.zeros(self.N,dtype=np.int) * np.nan
        for sig, idx in self.library.lookup.items():
            if sig in library.index:
                m[idx] = library.index[sig]
        return m
    def wrap(self,v):
        R""" wrap vectors into the periodic simulation box (respecting non-periodic directions)

        Args:
            v (array): array of vectors to wrap into the box

        Returns:
            w (array): array of wrapped vectors
        """
        pbc = np.asarray([dim in self.pbc for dim in 'xyz'],dtype=np.float)
        w = v - self.box * np.round( v / self.box * pbc)
        return w
    def save(self,filename,neighbors=False,library=False):
        R""" save info from the Snapshot as a pickle binary

        Args:
            filename (str): filename to save as
            neighbors (bool,optional): save neighborlist? (default False)
            library (bool,optional): save GraphLibrary? (default False)
        """
        buff = {}
        if neighbors:
            buff['neighbors'] = self.neighbors
        if library:
            buff['library'] = self.library
        with open(filename,'wb') as fid:
            pickle.dump(buff,fid)
    def load(self,filename):
        R""" load pre-computed info from a pickle binary

        Args:
            filename (str): filename to load
        """
        with open(filename,'rb') as fid:
            buff = pickle.load(fid)
        if 'neighbors' in buff:
            self.neighbors = buff['neighbors']
            self.N = len(self.neighbors)
        if 'library' in buff:
            self.library = buff['library']

class Ensemble:
    R""" handles collection and combination of Snapshots into single DMap,
         including many operations in parallel (when enabled)

    Args:
        **kwargs to be passed to Snapshot::buildLibrary()
    """
    def __init__(self,**kwargs):
        # evaluate options
        self.options = kwargs
        # set default values
        self.filenames = []
        self.library = classifiers.GraphLibrary()
        self.graph_lookups = {}
        self.dists = None
        self.valid_cols = None
        self.valid_rows = None
        self.invalid_rows = np.array([])
        self.dmap = dmap.DMap()
        self.color_rotation = None
        self.comm, self.size, self.rank, self.master = parallel.info()
        self.p = parallel.ParallelTask()
    def insert(self,key,snap):
        R""" insert a Snapshot into the Ensemble, automatically checks if the GraphLibrary
             has been built and calls Snapshot::buildLibrary() if it has not

        Args:
            key (hashable): a hashable key identifying this Snapshot in the graph_lookups dictionary
            snap (Snapshot): the Snapshot object to insert
        """
        if snap.library is None:
            snap.buildLibrary(**self.options)
        self.library.collect(snap.library)
        self.graph_lookups[key] = snap.library.lookup
    def backmap(self,snapkey):
        R""" obtain a mapping between the particles in a Snapshot and the Ensemble-wide GraphLibrary

        Args:
            key (hashable): the hashable key identifying this Snapshot in the graph_lookups dictionary

        Returns:
            m (array): the index of each particle in the Ensemble-wide GraphLibrary
        """
        N = np.sum(np.asarray([len(val) for key, val in self.graph_lookups[snapkey].items()]))
        m = np.zeros(N,dtype=np.int) * np.nan
        for sig, idx in self.graph_lookups[snapkey].items():
            if sig in self.library.index:
                m[idx] = self.library.index[sig]
        return np.array(m,dtype=np.int)
    def collect(self):
        R""" query, obtain, and merge Ensembles constructed in parallel (using ParallelTask class) """
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
            for key, val in other.graph_lookups.items():
                if key in self.graph_lookups:
                    print('Warning: duplicate graph lookup key detected during Ensemble.collect')
                self.graph_lookups[key] = val
        print('ensemble graph collection complete, found %d unique graphs'%len(self.library.items))
    def prune(self,freq_top=None,freq_thresh=None,freq_pct=None,
                   size_top=None,size_thresh=None,size_pct=None,
                   random=None):
        R""" choose landmark signatures for building the DMap based on various relevance metrics

        Args:
            freq_top (int,optional): number of most frequent signatures
            freq_thresh (int,optional): frequency threshold
            freq_pct (int,optional): frequency percentile threshold
            size_top (int,optional): number of largest clusters with same signature
            size_thresh (int,optional): cluster size threshold
            size_pct (int,optional): cluster size percentile threshold
            random (int,optional): number of random signatures (helps fill sample space)
        """
        if not self.master:
            return
        self.lm_idx = np.array([],dtype=np.int)
        if freq_top is not None:
            self.lm_idx = np.hstack((self.lm_idx,np.sort(np.argsort(self.library.counts)[::-1][:freq_top]).flatten()))
        if freq_thresh is not None:
            self.lm_idx = np.hstack((self.lm_idx,np.argwhere(self.library.counts >= freq_thresh).flatten()))
        if freq_pct is not None:
            self.lm_idx = np.hstack((self.lm_idx,np.argwhere(self.library.counts >= np.percentile(self.library.counts,freq_pct)).flatten()))
        if size_top is not None:
            self.lm_idx = np.hstack((self.lm_idx,np.sort(np.argsort(self.library.sizes)[::-1][:freq_top]).flatten()))
        if size_thresh is not None:
            self.lm_idx = np.hstack((self.lm_idx,np.argwhere(self.library.sizes >= freq_thresh).flatten()))
        if size_pct is not None:
            self.lm_idx = np.hstack((self.lm_idx,np.argwhere(self.library.sizes >= np.percentile(self.library.counts,size_pct)).flatten()))
        self.lm_idx = np.unique(self.lm_idx)
        if random is not None:
            remaining = range(len(vals))
            for idx in self.lm_idx:
                remaining.remove(idx)
            random = np.random.choice(remaining,num_random)
            self.lm_idx = np.unique(np.hstack((self.lm_idx,random)))
        self.lm_idx = np.unique(self.lm_idx)
        n = len(self.library.sigs)
        m = len(self.lm_idx)
        self.lm_sigs = [self.library.sigs[idx] for idx in self.lm_idx]
        print('using %d archetypal graphs as landmarks for %d less common ones'%(m,n-m))
    def computeDists(self):
        R""" compute distances between graphlet signatures, using landmarks if Ensemble.lm_idx has been set """
        if not self.master:
            return
        # prepare NGDVs for distance calculation
        dat = []
        for g in self.library.items:
            dat.append(g.ngdv)
        dat = np.array(dat)
        # perform distance calculation
        n = len(self.library.sigs)
        try:
            m = len(self.lm_sigs)
        except:
            m = n
            self.lm_idx = np.arange(n)
        self.dists = np.zeros((n,m))
        for i, lm in enumerate(self.lm_idx):
            self.dists[:,i] = np.linalg.norm(dat-dat[lm],axis=1)
    def detectDistOutliers(self,mode=None,thresh=None):
        R""" detect outliers in the distance matrix using either agglomerative
             clustering or simple cutoff

        Args:
            mode (str): 'agglomerative' or 'cutoff'
            thresh (float): threshold for clustering or cutoff
        """
        if self.master:
            # detect outliers, if requested
            if mode == 'agglomerative':
                # filter outliers such as vapor particles
                # first find bad landmarks
                d = np.sum(self.dists,axis=0)
                X = d.reshape(-1,1)
                #
                Z = hierarchy.linkage(X,'centroid')
                c = hierarchy.fcluster(Z,np.median(d),criterion='distance')
                c_med = [np.median(d[c==i]) for i in np.unique(c)]
                c_best = int(np.unique(c)[np.argwhere(c_med == np.min(c_med))])
                good_col = np.argwhere(c == c_best).flatten()
                bad_col = np.argwhere(c != c_best).flatten()
                self.lm_idx = self.lm_idx[good_col]
                self.valid_cols = good_col
                # then find other bad items
                d = np.sum(self.dists,axis=1)
                X = d.reshape(-1,1)
                Z = hierarchy.linkage(X,'centroid')
                c = hierarchy.fcluster(Z,np.median(d),criterion='distance')
                c_med = [np.median(d[c==i]) for i in np.unique(c)]
                c_best = int(np.unique(c)[np.argwhere(c_med == np.min(c_med))])
                self.valid_rows = np.argwhere(c == c_best).flatten()
                self.invalid_rows = np.argwhere(c != c_best).flatten()
            elif mode == 'cutoff':
                self.valid_cols = np.arange(self.dists.shape[1])
                d = np.min(self.dists,axis=1)
                self.valid_rows = np.argwhere(d < thresh).flatten()
                self.invalid_rows = np.argwhere(d >= thresh).flatten()
    def writeColors(self):
        R""" write color of each particle to file, in the order they appear in each Snapshot
             column 1 is signature index, columns 2-4 are RGB values
        """
        # share data among workers
        frame_maps = None
        color_coords = None
        if self.master:
            color_coords = np.copy(self.dmap.color_coords)
            frame_maps = {}
            for key in self.graph_lookups.keys():
                frame_maps[key] = self.backmap(key)
        color_coords = self.p.shareData(color_coords)
        frame_maps = self.p.shareData(frame_maps)
        # map local structure indices to ensemble
        keys = frame_maps.keys()
        keys.sort() # keys are not guaranteed to appear in the same order
        local_file_idx  = parallel.partition(range(len(keys)))
        for f in local_file_idx:
            filename = keys[f]
            fm = frame_maps[keys[f]].reshape(-1,1)
            cc = color_coords[fm,np.array([1,2,3])]
            if self.color_rotation is not None:
                if type(self.color_rotation) == tuple:
                    self.color_rotation = [self.color_rotation]
                for rot in self.color_rotation:
                    cc = util.rotate(cc,rot[0],rot[1])
            # need to distribute invalid_rows across ranks
            # for inv in self.invalid_rows:
            #     fm[fm==inv] = -1
            fdat = np.hstack((fm,cc))
            np.savetxt(filename + '.cmap', fdat)
    def buildDMap(self):
        R""" builds the diffusion map from pre-computed distances (computes them if necessary) """
        if self.master:
            if self.dists is None:
                if self.lm_idx is None:
                    print('Warning: computing pairwise distance matrix between ALL signatures')
                self.computeDists()
            self.dmap.build(self.dists,landmarks=self.lm_idx,
                            valid_cols=self.valid_cols,
                            valid_rows=self.valid_rows)
            print('Diffusion map construction complete')
    def makeSnapshot(self,filename):
        R""" create a Snapshot (in XYZ format) containing the low-dimensional manifold
             obtained from diffusion maps, along the corresponding color map

        Args:
            filename (str): filename to save as
        """
        if not self.master:
            return
        cd = np.copy(self.dmap.coords[:,1:4])
        cc = np.copy(self.dmap.color_coords[:,1:4])
        if self.color_rotation is not None:
            if type(self.color_rotation) == tuple:
                self.color_rotation = [self.color_rotation]
            for rot in self.color_rotation:
                cc = util.rotate(cc,rot[0],rot[1])
        # find bounds
        box = 2.*np.max(np.abs(cd),axis=0)
        snap = Snapshot()
        snap.box = box
        snap.xyz = cd
        snap.N = len(snap.xyz)
        io.writeXYZ('%s'%filename,snap)
        fm = np.arange(snap.N).reshape(-1,1)
        fdat = np.hstack((fm,cc))
        np.savetxt('%s.cmap'%filename,fdat)
