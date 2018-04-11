#
# nga.py
# define classes implementing Neighborhood Graph Analysis
#     for molecular simulation snapshots
#
# Copyright (c) 2018 Wesley Reinhart.
# This file is part of the crayon project, released under the Modified BSD License.

from __future__ import print_function

from crayon import _crayon
from crayon import bondorder
from crayon import classifiers
from crayon import parallel
from crayon import neighborlist
from crayon import color
from crayon import io
from crayon import dmap

import numpy as np
from scipy.cluster import hierarchy

try:
    import pickle
    allow_binary = True
except:
    allow_binary = False

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
        self.graph_library = None
        self.pattern_library = None
        # default options for building libraries
        self.pattern_mode = False
        self.global_mode = False
        self.graphlet_k = 5
        self.cluster = True
        self.cluster_thresh = None
        self.n_shells = 1
        self.q_thresh = None
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
        if self.pattern_mode:
            self.neighbors, self.all_neighbors = self.nl.getNeighbors(self)
        else:
            self.same_neighbors, self.neighbors = self.nl.getNeighbors(self)
    def buildAdjacency(self):
        if self.global_mode:
            self.adjacency = neighborlist.Network(self,k=self.graphlet_k)
        else:
            self.adjacency = _crayon.buildGraphs(self.neighbors,self.n_shells)
    def parseOptions(self,options):
        # use multi-atom Patterns?
        if 'pattern_mode' in options:
            self.pattern_mode = options['pattern_mode']
        # use a global Network instead of local Neighborhoods?
        if 'global_mode' in options:
            self.global_mode = options['global_mode']
        # choose size of graphlets
        if 'graphlet_k' in options:
            self.graphlet_k = options['graphlet_k']
        # perform clustering to find relevant structures?
        if 'cluster' in options:
            self.cluster = options['cluster']
        # threshold radius to use in the clustering
        if 'cluster_thresh' in options:
            self.cluster_thresh = options['cluster_thresh']
        # number of neighbor shells to use (forced)
        if 'n_shells' in options:
            self.n_shells = options['n_shells']
        # threshold for spherical harmonics
        if 'q_thresh' in options:
            self.q_thresh = options['q_thresh']
    def buildLibrary(self,**kwargs):
        self.parseOptions(kwargs)
        self.graph_library = classifiers.GraphLibrary()
        if self.adjacency is None:
            if self.neighbors is None:
                self.buildNeighborhoods()
            self.buildAdjacency()
        if self.q_thresh is not None:
            q_range = bondorder.computeQRange(self)
            disordered = np.argwhere(q_range < self.q_thresh).flatten()
            for idx in disordered:
                self.adjacency[idx] = np.ones((1,1))
        self.graph_library.build(self.adjacency,k=self.graphlet_k)
        if self.cluster:
            self.graph_library.sizes = neighborlist.largest_clusters(self,self.graph_library,self.cluster_thresh)
        if self.pattern_mode:
            self.pattern_library = classifiers.PatternLibrary()
            self.map_graphs = self.mapTo(self.graph_library)
            self.pattern_library.build(self.all_neighbors,
                                       self.graph_library.sigs,
                                       self.map_graphs)
    def mapTo(self,library):
        if type(library) == type(classifiers.GraphLibrary()):
            snap_lib = self.graph_library
        elif type(library) == type(classifiers.PatternLibrary()):
            snap_lib = self.pattern_library
        else:
            raise ValueError('must supply either a GraphLibrary or PatternLibrary object')
        m = np.zeros(self.N,dtype=np.int) * np.nan
        for sig, idx in snap_lib.lookup.items():
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
            buff['graph_library'] = self.graph_library
            buff['pattern_library'] = self.pattern_library
        with open(filename,'wb') as fid:
            pickle.dump(buff,fid)
    def load(self,filename):
        with open(filename,'rb') as fid:
            buff = pickle.load(fid)
        if 'neighbors' in buff:
            self.neighbors = buff['neighbors']
            self.N = len(self.neighbors)
        if 'graph_library' in buff:
            self.graph_library = buff['graph_library']
        if 'pattern_library' in buff:
            self.pattern_library = buff['pattern_library']

class Ensemble:
    def __init__(self,**kwargs):
        # evaluate optoins
        self.options = kwargs
        if 'pattern_mode' in self.options:
            self.pattern_mode = self.options['pattern_mode']
        else:
            self.pattern_mode = False
        # set default values
        self.filenames = []
        self.graph_library = classifiers.GraphLibrary()
        self.graph_lookups = {}
        self.graphs = []
        self.pattern_library = classifiers.PatternLibrary()
        self.pattern_lookups = {}
        self.patterns = []
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
        if self.pattern_mode:
            print('rank %d tasks complete, found %d unique graphs and %d unique patterns'%(self.rank,len(self.graph_library.items),len(self.pattern_library.items)))
        else:
            print('rank %d tasks complete, found %d unique graphs'%(self.rank,len(self.graph_library.items)))
        self.collect()
    def insert(self,idx,snap):
        if snap.graph_library is None:
            snap.buildLibrary(**self.options)
        self.graph_library.collect(snap.graph_library)
        self.graph_lookups[idx] = snap.graph_library.lookup
        if self.pattern_mode:
            self.pattern_library.collect(snap.pattern_library)
            self.pattern_lookups[idx] = snap.pattern_library.lookup
    def backmap(self,idx):
        N = 0
        for sig, idx in self.graph_lookups[idx].items():
            N += len(val)
        m = np.zeros(N,dtype=np.int) * np.nan
        for sig, idx in self.graph_lookups[idx].items():
            if sig in self.graph_library.index:
                m[idx] = self.graph_library.index[sig]
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
            self.graph_library.collect(other.graph_library)
            for key, val in other.graph_lookups.items():
                if key in self.graph_lookups:
                    print('Warning: duplicate graph lookup key detected during Ensemble.collect')
                self.graph_lookups[key] = val
        print('ensemble graph collection complete, found %d unique graphs'%len(self.graph_library.items))
        if self.pattern_mode:
            print('collecting patterns')
            # repeat with patterns
            for other in others:
                self.pattern_library.collect(other.pattern_library)
                for key, val in other.pattern_lookups.items():
                    if key in self.pattern_lookups:
                        print('Warning: duplicate pattern lookup key detected during Ensemble.collect')
                    self.pattern_lookups[key] = val
            print('ensemble pattern collection complete, found %d unique patterns'%len(self.pattern_library.items))
    def prune(self,num_random=None,num_top=None,min_freq=None,min_percentile=None,mode='frequency'):
        if not self.master:
            return
        if self.pattern_mode:
            library = self.pattern_library
        else:
            library = self.graph_library
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
    def getColorMaps(self,cidx):
        c, c_map = color.compressColors(self.dmap.color_coords[:,cidx],delta=0.001)
        if self.pattern_mode:
            lookups = self.pattern_lookups
            library = self.pattern_library
        else:
            lookups = self.graph_lookups
            library = self.graph_library
        frames = lookups.keys()
        frames.sort()
        frame_maps = []
        for f in frames:
            N = np.sum(np.asarray([len(val) for key, val in lookups[f].items()]))
            frame_data = np.zeros(N,dtype=np.int)
            for key, val in lookups[f].items():
                frame_data[np.asarray(val,dtype=np.int)] = library.index[key]
            frame_maps.append(frame_data)
        return c, c_map, frame_maps
    def computePatternDists(self):
        # prepare task list
        if self.master:
            n = len(self.pattern_library.sigs)
            try:
                m = len(self.lm_sigs)
            except:
                m = n
                self.lm_idx = np.arange(n)
            self.dists = np.zeros( (n,m) ) + np.Inf # designate null values with Inf
            for i in range(n):
                for j in range(m):
                    task_list.append( (i,self.lm_idx[j]) )
        # perform graph matching in parallel using MPI
        items = self.p.shareData(self.pattern_library.items)
        eval_func = lambda task, data: data[task[0]] - data[task[1]]
        result_list = self.p.computeQueue(function=eval_func,
                                          tasks=task_list,
                                          reports=10)
        # convert results into numpy array
        if self.master:
            for k, d in enumerate(result_list):
                i, j = task_list[k]
                jid = np.argwhere(self.lm_idx == j)[0]
                self.dists[i,jid] = d
    def computeGraphDists(self):
        if not self.master:
            return
        # prepare NGDVs for distance calculation
        dat = []
        for g in self.graph_library.items:
            dat.append(g.ngdv)
        dat = np.array(dat)
        # perform distance calculation
        n = len(self.graph_library.sigs)
        try:
            m = len(self.lm_sigs)
        except:
            m = n
            self.lm_idx = np.arange(n)
        self.dists = np.zeros((n,m))
        for i, lm in enumerate(self.lm_idx):
            self.dists[:,i] = np.linalg.norm(dat-dat[lm],axis=1)
    def computeDists(self):
        if self.pattern_mode:
            self.computePatternDists()
        else:
            self.computeGraphDists()
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
    def autoColor(self,prefix='draw_colors',sigma=1.0,VMD=False,Ovito=False,similarity=True):
        coms = None
        if self.master:
            coms, best = self.dmap.uncorrelatedTriplets()
            print('probable best eigenvector triplet is %s'%str(coms[best]))
        coms = self.p.shareData(coms)
        self.colorTriplets(coms,prefix=prefix,sigma=sigma,VMD=VMD,Ovito=Ovito,similarity=similarity)
    def colorTriplets(self,trips,prefix='draw_colors',sigma=1.0,
                      VMD=False,Ovito=False,similarity=True,
                      rotation=None):
        # enforce list-of-lists style triplets
        if type(trips[0]) == int:
            trips = [trips]
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
                for inv in self.invalid_rows:
                    mapped_color[mapped_color==inv] = -1
                f_dat = np.hstack((mapped_color,sim))
                np.savetxt(filename + '_%d%d%d.cmap'%trip, f_dat)
        if not self.master:
            return
        # write visualization scripts
        for t, trip in enumerate(trips):
            if VMD:
                trip_colors = colors[t]
                if rotation is not None:
                    trip_colors = color.rotate(trip_colors,rotation[0],rotation[1])
                color.writeVMD('%s_%d%d%d.tcl'%(prefix,trip[0],trip[1],trip[2]),
                               self.filenames, trip_colors, trip, f_dat.shape[1], sigma=sigma,
                               swap=('/home/wfr/','/Users/wfr/mountpoint/'))
    def buildDMap(self):
        if self.master:
            self.dmap.build(self.dists,landmarks=self.lm_idx,
                            valid_cols=self.valid_cols,
                            valid_rows=self.valid_rows)
            print('Diffusion map construction complete')
            self.dmap.write()
            np.savetxt('graph-counts.dat',self.graph_library.counts)
            np.savetxt('graph-sizes.dat',self.graph_library.sizes)
