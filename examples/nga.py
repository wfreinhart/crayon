from __future__ import print_function
import crayon

# snapshots to analyze
filenames = ['bcc.xyz','fcc.xyz','hcp.xyz','liquid.xyz']

# build neighbor list with a Voronoi construction
#   (or equivalenty Delaunay triangulation)
nl = crayon.neighborlist.Voronoi()

# use one neighbor shell for neighborhood graphs
traj = crayon.nga.Ensemble()

# read snapshots and build neighborhoods in parallel
local_filenames = crayon.parallel.partition(filenames)
for f in local_filenames:
    print('rank %d analyzing %s'%(traj.rank,f))
    xyz, box = crayon.io.readXYZ(f)
    snap = crayon.nga.Snapshot(xyz=xyz,box=box,nl=nl,pbc='xyz')
    traj.insert(f,snap)

# merge trajectories from different ranks
traj.collect()

# define landmarks as signatures with large clusters
traj.prune(size_pct=90, random=100)

# compute distances between neighborhood graphs
traj.computeDists()

# detect outliers using agglomerative clustering
traj.detectOutliers(mode='agglomerative')

# take distances to a power <1 to exaggerate polymorphs
traj.dmap.alpha = 0.50

# build diffusion map from distance matrix
traj.buildDMap()

# write color maps for visualization in Ovito or VMD
traj.writeColors()

# create a "snapshot" of the manifold for easy visualization
traj.makeSnapshot('manifold.xyz')
