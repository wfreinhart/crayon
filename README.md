# crayon

crayon is a python plugin for performing graphlet decomposition and comparisons between networks.
It wraps the libgraphlet library, which in turn wraps [Orca](http://www.biolab.si/supp/orca/orca.html),
based on the following [paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2623288/):
Milenkoviæ, Tijana, and Nataša Pržulj. "Uncovering biological network function via graphlet degree signatures."
Cancer informatics 6 (2008): 257.

crayon is released under the Modified BSD License. Please read the [license](LICENSE.md) for exact terms.

## Compiling crayon

To build crayon, ensure that the `hoomd` module is on your Python path
(or hint to its location using `HOOMD_ROOT`), and install to an appropriate location:

```bash
cd /path/to/crayon
mkdir build && cd build
cmake ..
make install
```

You must make sure that your installation location is on your `PYTHONPATH`, and then `crayon` can
be imported as usual

```python
import crayon
```

### Prerequisites

 * Required:
     * Python >= 2.7
     * numpy >= 1.7
     * CMake >= 3.1.0
     * C++11 compliant compiler
     * Boost graph library headers
 * Included (as git submodules):
     * Eigen (header only)
     * pybind11 (header only)
     * libgraphlet (linked automatically)

### Testing

All code is unit-tested at the Python level. To run all tests from the build directory,

```bash
make test
```