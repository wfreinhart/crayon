#ifndef SRC_COMPARISON_H_
#define SRC_COMPARISON_H_

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include <Eigen/Core>

#include "PyGraph.h"

namespace crayon
{

Eigen::MatrixXd GDVSimilarity(PyGraph &A, PyGraph &B);
Eigen::MatrixXd GDDAgreement(PyGraph &A, PyGraph &B);

void export_Comparison(pybind11::module& m);

} // end namespace crayon

#endif // SRC_PY_GRAPH_H_
