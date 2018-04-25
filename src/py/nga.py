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
from crayon import color
from crayon import io
from crayon import dmap

import numpy as np
from scipy.cluster import hierarchy
import pickle

class Snapshot:
    R""" holds necessary data from a simulation snapshot and handles
         neighborlist generation and graph library construction

    Args:
        reader_input (tuple): the input tuple for the reader function
        reader (function): takes (Snapshot,reader_input) as input and
                           sets Snapshot.N, Snapshot.L, and Snapshot.xyz
        nl (crayon::Neighborlist): a neighborlist generation class
        pbc (str) (optional): dimensions with periodic boundaries (defaults to 'xyz')
    """
    def __init__(self,reader_input,reader=None,nl=None,pbc='xyz'):
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
        # auto-detect 2D configuration
        span = np.max(self.xyz,axis=0) - np.min(self.xyz,axis=0)
        dims = 'xyz'
        for i, s in enumerate(span):
            if s < 1e-4:
                print('detected 2D configuration')
                # force values for better compatibility with Voro++
                self.L[i] = 1.
                self.xyz[:,i] = 0.
                self.pbc = self.pbc.replace(dims[i],'')
    def buildNeighborhoods(self):
        self.neighbors = self.nl.getNeighbors(self)
    def buildAdjacency(self):
        if self.global_mode:
            self.adjacency = neighborlist.Network(self,k=self.graphlet_k)
        else:
            self.adjacency = _crayon.buildGraphs(self.neighbors,self.n_shells)
    def parseOptions(self,options):
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
        m = np.zeros(self.N,dtype=np.int) * np.nan
        for sig, idx in self.library.lookup.items():
            if sig in library.index:
                m[idx] = library.index[sig]
        return m
    def wrap(self,v):
        pbc = np.asarray([dim in self.pbc for dim in 'xyz'],dtype=np.float)
        w = v - self.L * np.round( v / self.L * pbc)
        return w
    def save(self,filename,neighbors=False,library=False):
        buff = {}
        if neighbors:
            buff['neighbors'] = self.neighbors
        if library:
            buff['library'] = self.library
        with open(filename,'wb') as fid:
            pickle.dump(buff,fid)
    def load(self,filename):
        with open(filename,'rb') as fid:
            buff = pickle.load(fid)
        if 'neighbors' in buff:
            self.neighbors = buff['neighbors']
            self.N = len(self.neighbors)
        if 'library' in buff:
            self.library = buff['library']

class Ensemble:
    def __init__(self,**kwargs):
        # evaluate optoins
        self.options = kwargs
        # set default values
        self.filenames = []
        self.library = classifiers.GraphLibrary()
        self.graph_lookups = {}
        self.graphs = []
        self.dists = None
        self.valid_cols = None
        self.valid_rows = None
        self.invalid_rows = np.array([])
        self.dmap = dmap.DMap()
        self.comm, self.size, self.rank, self.master = parallel.info()
        self.p = parallel.ParallelTask()
    def neighborhoodsFromFile(self,filenames,nl):
        if type(filenames) == str:
            filenames = [filenames]
        self.filenames = filenames
        local_file_idx  = parallel.partition(range(len(self.filenames)))
        for f in local_file_idx:
            filename = self.filenames[f]
            print('rank %d of %d will process %s'%(self.rank,self.size,filename))
            # create snapshot instance and build neighborhoods
            snap = Snapshot(filename,pbc='xyz',nl=nl)
            self.insert(f,snap)
            snap.save(filename + '.nga',neighbors=True)
        print('rank %d tasks complete, found %d unique graphs'%(self.rank,len(self.library.items)))
        self.collect()
    def insert(self,idx,snap):
        if snap.library is None:
            snap.buildLibrary(**self.options)
        self.library.collect(snap.library)
        self.graph_lookups[idx] = snap.library.lookup
    def backmap(self,idx):
        N = 0
        for sig, idx in self.graph_lookups[idx].items():
            N += len(val)
        m = np.zeros(N,dtype=np.int) * np.nan
        for sig, idx in self.graph_lookups[idx].items():
            if sig in self.library.index:
                m[idx] = self.library.index[sig]
        return np.array(m,dtype=np.int)
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
            for key, val in other.graph_lookups.items():
                if key in self.graph_lookups:
                    print('Warning: duplicate graph lookup key detected during Ensemble.collect')
                self.graph_lookups[key] = val
        print('ensemble graph collection complete, found %d unique graphs'%len(self.library.items))
    def prune(self,num_random=None,num_top=None,min_freq=None,min_percentile=None,mode='frequency'):
        if not self.master:
            return
        library = self.library
        if mode == 'frequency':
            vals = library.counts
        elif mode == 'clustersize':
            vals = library.sizes
        else:
            raise ValueError('must specify a valid mode (frequency of clustersize)')
        self.lm_idx = np.array([])
        if num_top is not None:
            self.lm_idx = np.sort(np.argsort(vals)[::-1][:num_top]).flatten()
        elif min_freq is not None:
            self.lm_idx = np.argwhere(vals >= min_freq).flatten()
        elif min_percentile is not None:
            self.lm_idx = np.argwhere(vals >= np.percentile(vals,min_percentile)).flatten()
        else:
            raise RuntimeError('must supply either num_landmarks or min_freq')
        if num_random is not None:
            remaining = range(len(vals))
            for idx in self.lm_idx:
                remaining.remove(idx)
            random = np.random.choice(remaining,num_random)
            self.lm_idx = np.unique(np.hstack((self.lm_idx,random)))
        n = len(library.sigs)
        m = len(self.lm_idx)
        self.lm_sigs = [library.sigs[idx] for idx in self.lm_idx]
        print('using %d archetypal graphs as landmarks for %d less common ones'%(m,n-m))
    def getFrameMaps(self,cidx):
        lookups = self.graph_lookups
        library = self.library
        frames = lookups.keys()
        frames.sort()
        frame_maps = []
        for f in frames:
            N = np.sum(np.asarray([len(val) for key, val in lookups[f].items()]))
            frame_data = np.zeros(N,dtype=np.int)
            for key, val in lookups[f].items():
                frame_data[np.asarray(val,dtype=np.int)] = library.index[key]
            frame_maps.append(frame_data)
        return frame_maps
    def computeDists(self):
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
    def writeColors(self,bonds=False,rotation=None):
        trip = np.array([1,2,3])
        # share data among workers
        frame_maps = None
        color_coords = None
        if self.master:
            color_coords = self.dmap.color_coords
            frame_maps = self.getFrameMaps(trip)
        color_coords = self.p.shareData(color_coords)
        frame_maps = self.p.shareData(frame_maps)
        # map local structure indices to ensemble
        local_file_idx  = parallel.partition(range(len(self.filenames)))
        for f in local_file_idx:
            filename = self.filenames[f]
            if bonds:
                nl = neighborlist.NeighborList()
                snap = Snapshot(filename,nl=nl)
                snap.load(filename + '.nga')
                filetype = filename[::-1].find('.')
                io.writeXML(filename[:-filetype] + 'bonds.xml',snap,bonds=bonds)
            else:
                snap = Snapshot(filename + '.nga')
            fm = frame_maps[f].reshape(-1,1)
            cc = color_coords[fm,trip]
            if rotation is not None:
                if type(rotation) == tuple:
                    rotation = [rotation]
                for rot in rotation:
                    cc = color.rotate(cc,rot[0],rot[1])
            for inv in self.invalid_rows:
                fm[fm==inv] = -1
            f_dat = np.hstack((fm,cc))
            np.savetxt(filename + '.cmap', f_dat)
    def buildDMap(self,freq=None):
        if self.master:
            self.dmap.build(self.dists,landmarks=self.lm_idx,
                            valid_cols=self.valid_cols,
                            valid_rows=self.valid_rows,
                            freq=freq)
            print('Diffusion map construction complete')
