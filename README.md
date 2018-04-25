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

Snapshots can be analyzed individually or as a trajectory. To build a `Library` of all unique
graphs in a simulation `Snapshot`:

```python
nl = crayon.neighborlist.Voronoi()
reader_input = 'my_file.xyz'
snap = crayon.nga.Snapshot(reader_input,reader=crayon.io.readXYZ,nl=nl)
snap.buildLibrary()
```

The `reader` argument can be a reference to any function which accepts `(snap,reader_input)` as
input and sets the `snap.N` (number of particles), `snap.L` (box dimensions), `snap.xyz` (positions),
and optionally `snap.T` (types). `crayon.io` includes functions for reading XYZ, XML, and GSD files
and will automatically use the appropriate function if no `reader` is supplied.

NGA operates on `Ensembles` of `Snapshots`. Ensembles can be assembled from Snapshots or generated
from a list of file paths:

```python
ens = crayon.nga.Ensemble()
ens.insert(0,snap)
my_files = ['my_file_1.xyz','my_file_2.xyz']
ens.neighborhoodsFromFile(my_files,nl)
```

Once an `Ensemble` is provided with `Snapshots`, a `DMap` can be computed to provide a low-dimensional
representation of all observed structures in the `Ensemble`.

```python
ens.computeDists()
ens.buildDMap()
```

The coordinates in low-dimensional space are stored in `ens.dmap.color_coords`, and can be written to
files using the following convenience function:

```python
ens.colorTriplets((1,2,3))
```

A `.cmap` file will be written for each `Snapshot` in the `Ensemble` containing a list of structure ID
and RGB triplet for each particle.