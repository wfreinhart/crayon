#
# neighborlist.py
# build neighbor lists from particle positions
#
# Copyright (c) 2018 Wesley Reinhart.
# This file is part of the crayon project, released under the Modified BSD License.

from __future__ import print_function
import sys

import numpy as np

from crayon import _crayon

from scipy.cluster import hierarchy

def visit(i,snap,particles,visited,members,level,remaining):
    if level >= sys.getrecursionlimit()/2:
        return False
    idx = int(np.argwhere(particles==i))
    if visited[idx] == 1:
        return True
    members.append(i)
    visited[idx] = 1
    nn = [x for x in snap.neighbors[i] if x in particles]
    for j in nn:
        jdx = np.argwhere(particles==j)
        if visited[jdx] == 0:
            result = visit(j,snap,particles,visited,members,level+1,remaining)
            if not result:
                remaining += [j]
    return True

def largest_clusters(snap,library,thresh=None):
    sizes = []
    if thresh is not None:
        dat = []
        for g in library.items:
            dat.append(g.ngdv)
        dat = np.array(dat)
        n = len(library.items)
        D = np.zeros((n,n))
        for i in range(n):
            D[:,i] = np.linalg.norm(dat-dat[i],axis=1)
    for i, sig in enumerate(library.sigs):
        # include exact matches
        particles = np.copy( library.lookup[sig].flatten() )
        if thresh is not None:
            # include all particles within a threshold radius
            for j in np.argwhere(D[i,:]<thresh).flatten():
                new_sig = library.sigs[j]
                new_particles = library.lookup[new_sig].flatten()
                particles = np.hstack((particles,new_particles))
        particles = np.unique(particles)
        visited = np.zeros(len(particles))
        largest = 0
        while np.sum(visited) < len(particles):
            remaining = [particles[np.argwhere(visited == 0).flatten()[0]]]
            members = []
            while len(remaining) > 0:
                root = remaining.pop()
                result = visit(root,snap,particles,visited,members,1,remaining)
            largest = max(largest,len(members))
        sizes.append(largest)
    return np.array(sizes)

def shell(i,NL,n):
    idx = NL[i].flatten()
    s = 1
    while s < n:
        shell2 = []
        for j in range(len(idx)):
            shell2 += list(NL[idx[j]])
        shell2 = np.unique(np.array(shell2,dtype=np.int))
        idx = np.array(shell2)
        s += 1
    return idx

# builds an adjacency matrix from the nearest neighbor list
def particleAdjacency(i, NL, n=1):
    idx = shell(i,NL,n)
    idx = np.hstack(([i],np.sort(idx[idx!=i]))) # enforce deterministic ordering
    n = len(idx)
    A = np.zeros((n,n),np.int8)
    for j in range(len(idx)):
        for k in range(len(idx)):
            A[j,k] = int( (idx[k] in NL[idx[j]].flatten()) or j == k )
    # enforce symmetry
    for j in range(len(idx)-1):
        for k in range(j+1,len(idx)):
            if A[j,k] == 1 or A[k,j] == 1:
                A[j,k] = 1
                A[k,j] = 1
    return A

class Network:
    def __init__(self,snap,k=5):
        A = np.zeros((snap.N,snap.N),np.int8)
        for i, nn in enumerate(snap.neighbors):
            A[i,nn] = 1
        self.cpp = _crayon.neighborhood(A,k)
        self.gdv = self.cpp.gdv()
        # weight GDVs according to dependencies between orbits
        o = np.array([1, 2, 2, 2, 3, 4, 3, 3, 4, 3,
                      4, 4, 4, 4, 3, 4, 6, 5, 4, 5,
                      6, 6, 4, 4, 4, 5, 7, 4, 6, 6,
                      7, 4, 6, 6, 6, 5, 6, 7, 7, 5,
                      7, 6, 7, 6, 5, 5, 6, 8, 7, 6,
                      6, 8, 6, 9, 5, 6, 4, 6, 6, 7,
                      8, 6, 6, 8, 7, 6, 7, 7, 8, 5,
                      6, 6, 4],dtype=np.float)
        w = 1. - o / 73.
        self.ngdv = self.gdv * w[:self.gdv.shape[1]]
        ones = np.ones(snap.N)
        sums = np.sum(self.ngdv,axis=1)
        norm = np.reshape(np.max(np.vstack((sums,ones)),axis=0),(-1,1))
        self.ngdv = self.ngdv / norm
        self.graphs = []
        for i in range(snap.N):
            self.graphs.append( (self.gdv[i],self.ngdv[i]) )
    def __iter__(self):
        self.i = -1
        return self
    def __len__(self):
        return len(self.ngdv)
    def next(self):
        self.i += 1
        if self.i >= len(self.ngdv):
            raise StopIteration
        return (self.gdv[self.i],self.ngdv[self.i])

class NeighborList:
    def __init__(self,enforce_symmetry=True,max_neighbors=None):
        self.enforce_symmetry = enforce_symmetry
        self.max_neighbors = max_neighbors
        self.setParams()
    def setParams(self):
        pass
    def getNeighbors(self,snap):
        return [], []
    def symmetrize(self,NL):
        for i, nn in enumerate(NL):
            for j in nn:
                if i not in NL[j]:
                    NL[j] = np.append(NL[j],i)
    def removeOverbonded(self,NL):
        for i, nn in enumerate(NL):
            if len(nn) > self.max_neighbors+1:
                NL[i] = np.array([i])
                for j in nn:
                    NL[j] = np.delete(NL[j],np.argwhere(NL[j]==i))
    def getAdjacency(self,snap):
        adjacency = []
        for i in range(snap.N):
                adjacency.append(self.particleAdjacency(i,snap.neighbors))
        return adjacency

class CellList(NeighborList):
    def setParams(self,rcut=None):
        if rcut is not None:
            self.rcut = rcut
    def getNeighbors(self,snap):
        cell = _crayon.cellpp(snap.xyz, snap.box, 'x' in snap.pbc, 'y' in snap.pbc, 'z' in snap.pbc, self.rcut)
        nl = []
        for i in snap.N:
            # sort cell list result
            cc = cell[i]
            dvec = snap.wrap(snap.xyz[cc] - snap.xyz[i])
            d = np.sqrt(np.sum(dvec**2.,axis=1))
            nl.append(cc[d<=self.rcut])
        return nl

class Hybrid(NeighborList):
    def setParams(self,rcut=None):
        if rcut is not None:
            self.rcut = rcut
    def getNeighbors(self,snap):
        # build all-atom neighborlist with Voro++
        voro = _crayon.voropp(snap.xyz, snap.box, 'x' in snap.pbc, 'y' in snap.pbc, 'z' in snap.pbc)
        cell = _crayon.cellpp(snap.xyz, snap.box, 'x' in snap.pbc, 'y' in snap.pbc, 'z' in snap.pbc, self.rcut)
        nl = []
        for i in snap.N:
            # sort voro result
            vv = np.hstack((i,np.array(voro[i])))
            dvec = snap.wrap(snap.xyz[vv] - snap.xyz[i])
            d = np.sqrt(np.sum(dvec**2.,axis=1))
            vv = vv[np.argsort(d)]
            # sort cell list result
            cc = cell[i]
            dvec = snap.wrap(snap.xyz[cc] - snap.xyz[i])
            d = np.sqrt(np.sum(dvec**2.,axis=1))
            cc = cc[np.argsort(d)]
            # compare
            n = min(len(vv),len(cc))
            ndiv = np.searchsorted(cc[:n] != vv[:n],True)
            nl.append(cc[:ndiv])
        return nl

class Voronoi(NeighborList):
    def setParams(self,r_max=None,r_max_multiple=None,
                  clustering=True,cluster_method='centroid',cluster_ratio=0.25):
        self.r_max = r_max
        self.r_max_multiple = r_max_multiple
        self.clustering = clustering
        self.cluster_method = cluster_method
        self.cluster_ratio = cluster_ratio
    # filter neighbors by hierarchical clustering
    def filterNeighbors(self,nl_idx,snap_idx,neighbors,snap):
        # get neighbors from triangulation
        nn = np.array(neighbors[nl_idx],dtype=np.int)
        n_voro = len(nn)
        # remove negative IDs (Voro++ indicating that a particle is its own neighbor)
        #    this line is problematic for type-specific operations
        nn[nn<0] = snap.N + nn[nn<0]
        # remove duplicates
        nn = np.unique(nn)
        # remove self
        nn = nn[nn!=snap_idx]
        # get displacement vectors and ask Snapshot to wrap them
        d_vec = snap.wrap(snap.xyz[nn,:] - snap.xyz[snap_idx,:])
        # sort neighbors by increasing distance
        d_nbr = np.sqrt(np.sum((d_vec)**2.,1))
        # find sorted order
        order = np.argsort(d_nbr)
        nn = nn[order]
        d_nbr = d_nbr[order]
        # apply maximum cutoff radius
        if self.r_max_multiple is not None:
            r_max = self.r_max_multiple * d_nbr[0]
        elif self.r_max is not None:
            r_max = self.r_max
        else:
            r_max = None
        if r_max is not None:
            nn = nn[d_nbr <= r_max]
            d_nbr = d_nbr[d_nbr <= r_max]
            if len(nn) < 2:
                nn = np.hstack(([snap_idx],nn))
                return nn
        # remove entries that will never appear in the first cluster
        thresh = self.cluster_ratio * d_nbr[0]
        delta = np.diff(d_nbr)
        valid = np.hstack((0,np.argwhere(delta < thresh).flatten()+1))
        # check if all entries will always belong to one cluster
        if d_nbr[valid[-1]] - d_nbr[valid[0]] < thresh:
            return np.hstack(([snap_idx],nn[valid]))
        # do scipy clustering method
        X = d_nbr.reshape(-1,1)
        Z = hierarchy.linkage(X,self.cluster_method)
        c = hierarchy.fcluster(Z,self.cluster_ratio*d_nbr[0],criterion='distance')
        h_base = np.argwhere(c == c[0]).flatten()
        f_open = n_voro / float(len(h_base))
        return np.hstack((snap_idx,nn[h_base]))
    # compute Delaunay triangulation with Voro++ library
    def getNeighbors(self,snap):
        # build all-atom neighborlist with Voro++
        nl = _crayon.voropp(snap.xyz, snap.box, 'x' in snap.pbc, 'y' in snap.pbc, 'z' in snap.pbc)
        all_neighbors = []
        for idx in range(snap.N):
            if self.clustering:
                nn = self.filterNeighbors(idx,idx,nl,snap)
            else:
                nn = nl[idx]
            all_neighbors.append(np.array(nn,dtype=np.int))
        if self.enforce_symmetry:
            self.symmetrize(all_neighbors)
        if self.max_neighbors is not None:
            self.removeOverbonded(all_neighbors)
        return all_neighbors
