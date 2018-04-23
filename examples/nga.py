from crayon import *

# snapshots to analyze
filenames = ['bcc.xml','fcc.xml','hcp.xml','p.xml','sc.xml','ct.xml','ht.xml','liquid.xml']

# build neighbor list with a Voronoi construction
#   (or equivalenty Delaunay triangulation)
nl = neighborlist.Voronoi()

# use one neighbor shell for neighborhood graphs
traj = nga.Ensemble(n_shells=1)

# read snapshots and build neighborhoods
traj.neighborhoodsFromFile(filenames,nl)

# define landmarks as signatures with large clusters
traj.prune(mode='clustersize',min_freq=3)

# compute distances between neighborhoods and landmarks
traj.computeDists()

# take all distances to power of 0.5
traj.dmap.alpha = 0.5

# build diffusion map from distance matrix
traj.buildDMap()

# create a VMD visualization script based on the first three eigenvectors
traj.colorTriplets([(1,2,3)],sigma=1.0,VMD=True)
