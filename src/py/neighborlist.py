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

try:
    import freud
    foundFreud = True
except:
    print('Warning: freud python module not found, neighborlist.AdaptiveCNA will not be available')
    foundFreud = False

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

def largest_clusters(snap,library):
    sizes = []
    for i, sig in enumerate(library.sigs):
        particles = library.lookup[sig]
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

class NeighborList:
    def __init__(self,second_shell=(0,0),enforce_symmetry=True):
        self.second_shell = second_shell
        self.enforce_symmetry = enforce_symmetry
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
    # builds an adjacency matrix from the nearest neighbor list
    def particleAdjacency(self,i, NL):
        idx = NL[i].flatten()
        if len(idx) <= self.second_shell[0]:
            shell2 = []
            for j in range(len(idx)):
                shell2 += list(NL[idx[j]])
            shell2 = np.unique(np.array(shell2,dtype=np.int))
            if len(shell2) <= self.second_shell[1]:
                idx = np.array(shell2)
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
    def getAdjacency(self,snap):
        adjacency = []
        for i in range(snap.N):
                adjacency.append(self.particleAdjacency(i,snap.neighbors))
        return adjacency

class AdaptiveCNA(NeighborList):
    def setParams(self,r_max=4.,max_nbr=16,
                  strict_rcut=True,near_nbr=6,geo_factor=1.2071):
        self.r_max = r_max
        self.max_nbr = max_nbr
        self.strict_rcut = strict_rcut
        self.near_nbr = near_nbr
        self.geo_factor = geo_factor
        if not foundFreud:
            raise RuntimeError('neighborlist.AdaptiveCNA requires freud')
    def getNeighbors(self,snap):
        box = freud.box.Box(Lx=snap.L[0],Ly=snap.L[1],Lz=snap.L[2],is2D=False)
        nl  = freud.locality.NearestNeighbors(self.r_max,self.max_nbr,strict_cut=self.strict_rcut)
        nl.compute(box,snap.xyz,snap.xyz)
        Rsq = nl.getRsqList()
        Rsq[Rsq < 0] = np.nan
        R6 = np.nanmean(Rsq[:,:self.near_nbr],axis=1)
        Rcut = self.geo_factor**2. * R6
        nl = nl.getNeighborList()
        neighbors = []
        for i in range(snap.N):
            neighbors.append(np.hstack(([i],nl[i,Rsq[i,:]<Rcut[i]])))
        return neighbors, neighbors

class Voronoi(NeighborList):
    def setParams(self,r_max=None,
                  clustering=True,cluster_method='centroid',cluster_ratio=0.25):
        self.r_max = r_max
        self.clustering = clustering
        self.cluster_method = cluster_method
        self.cluster_ratio = cluster_ratio
    # filter neighbors by hierarchical clustering
    def filterNeighbors(self,nl_idx,snap_idx,neighbors,snap):
        # get neighbors from triangulation
        nn = np.array(neighbors[nl_idx],dtype=np.int)
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
        order = np.argsort(d_nbr)
        nn = np.array(nn)[order]
        d_nbr = d_nbr[order]
        # apply maximum cutoff radius
        if self.r_max is not None:
            nn = nn[d_nbr <= self.r_max]
            d_nbr = d_nbr[d_nbr <= self.r_max]
            if len(nn) < 2:
                nn = np.hstack(([snap_idx],nn))
                return nn
        # exclude far-away particles by clustering
        X = d_nbr.reshape(-1,1)
        Z = hierarchy.linkage(X,self.cluster_method)
        c = hierarchy.fcluster(Z,self.cluster_ratio*d_nbr[0],criterion='distance')
        h_base = np.argwhere(c == c[0]).flatten()
        nn = np.hstack(([snap_idx],nn[h_base]))
        return nn
    # compute Delaunay triangulation with Voro++ library
    def getNeighbors(self,snap):
        # build all-atom neighborlist with Voro++
        nl, areas = _crayon.voropp(snap.xyz, snap.L,
                            'x' in snap.pbc, 'y' in snap.pbc, 'z' in snap.pbc)
        all_neighbors = []
        for idx in range(snap.N):
            if self.clustering:
                nn = self.filterNeighbors(idx,idx,nl,snap)
            else:
                nn = nl[idx]
            all_neighbors.append(np.array(nn,dtype=np.int))
        if self.enforce_symmetry:
            self.symmetrize(all_neighbors)
        if len(np.unique(snap.T)) == 1:
            return all_neighbors, all_neighbors
        # use neighborhood to build multi-atom patterns
        same_neighbors = [[] for idx in range(snap.N)]
        for t in np.unique(snap.T):
            t_idx = np.argwhere(snap.T==t).flatten()
            nl = _crayon.voropp(snap.xyz[t_idx,:], snap.L,
                                'x' in snap.pbc, 'y' in snap.pbc, 'z' in snap.pbc)
            for i in range(len(nl)):
                nl[i] = np.unique(t_idx[np.array(nl[i])])
            for i in range(len(t_idx)):
                if self.clustering:
                    nn = self.filterNeighbors(i,t_idx[i],nl,snap)
                else:
                    nn = nl[i]
                same_neighbors[t_idx[i]] = nn
        if self.enforce_symmetry:
            self.symmetrize(same_neighbors)
        return same_neighbors, all_neighbors
