//
// Comparison.h
// provides functions for pairwise comparison of GDVs and GDDs
//
// Copyright (c) 2018 Wesley Reinhart.
// This file is part of the crayon project, released under the Modified BSD License.

#ifndef SRC_COMPARISON_H_
#define SRC_COMPARISON_H_

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include <Eigen/Core>

#include "Neighborhood.h"

namespace crayon
{

Eigen::MatrixXd GDVSimilarity(Neighborhood &A, Neighborhood &B);
Eigen::VectorXd GDDAgreement(Neighborhood &A, Neighborhood &B);

void export_Comparison(pybind11::module& m);

} // end namespace crayon

#endif // SRC_PY_GRAPH_H_
