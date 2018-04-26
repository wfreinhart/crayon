# crayon

crayon is a python plugin for performing Neighborhood Graph Analysis (NGA) via graphlet decomposition
for autonomous crystal structure comparison on molecular simulation snapshots.
It wraps the libgraphlet library, which in turn wraps [Orca](http://www.biolab.si/supp/orca/orca.html),
based on the following [paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2623288/):
Milenkoviæ, Tijana, and Nataša Pržulj. "Uncovering biological network function via graphlet degree signatures."
Cancer informatics 6 (2008): 257.

crayon is released under the Modified BSD License. Please read the [license](LICENSE.md) for exact terms.

## Obtaining crayon

Clone from the git repository and initialize the submodules. You must do this recursively to ensure all
dependencies are obtained!

```bash
git clone https://github.com/wfreinhart/crayon.git
cd crayon
git submodule update --init --recursive
```

## Compiling crayon

To build crayon, simply use `cmake` to install to an appropriate location:

```bash
cd /path/to/crayon
mkdir build && cd build
cmake ..
make install
```

### Prerequisites

 * Required:
     * [PyDMap](https://github.com/awlong/DiffusionMap)
     * Python >= 2.7
     * numpy >= 1.7
     * scipy >= 1.0.0
     * CMake >= 3.1.0
     * C++11 compliant compiler
     * Boost graph library headers
 * Included (as git submodules):
     * Eigen (header only)
     * pybind11 (header only)
     * libgraphlet (built and linked automatically)
     * Voro++ (built and linked automatically)

### Testing

All code is unit-tested at the Python level. To run all tests from the build directory,

```bash
make test
```

## Usage

You must make sure that your installation location is on your `PYTHONPATH`, and then `crayon` can
be imported as usual:

```python
import crayon
```

Each simulation `Snapshot` must be initialiezd with particle positions, box dimensions, and a
reference to a `NeighborList` object:

```python
nl = crayon.neighborlist.Voronoi()
snap = crayon.nga.Snapshot(xyz=xyz,box=box,nl=nl)
```

NGA is performed on an `Ensemble`, which is a collection of `Snapshot` objects. An `Ensemble` is
be built up from a set of `Snapshot` objects using the `insert` method:

```python
traj = crayon.nga.Ensemble()
traj.insert('my_filename.xyz',snap)
```

Once an `Ensemble` is loaded with `Snapshot`s, a `DMap` can be computed to provide a low-dimensional
representation of all observed structures:

```python
traj.computeDists()
traj.buildDMap()
```

The coordinates in low-dimensional space can be written to files using the following convenience function:

```python
traj.writeColors()
```

A `.cmap` file will be written for each `Snapshot` in the `Ensemble` containing the structure ID
and RGB triplet for each particle. The classification can be easily visualized in [Ovito](http://www.ovito.org/)
using the `Python script` modifier with the script located in `src/py/ovito.py`.
