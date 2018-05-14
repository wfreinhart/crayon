//
// Neighbors.h
// wraps the voro++ library for calculation of neighbor lists
//
// Copyright (c) 2018 Wesley Reinhart.
// This file is part of the crayon project, released under the Modified BSD License.

#ifndef SRC_NEIGHBORS_H_
#define SRC_NEIGHBORS_H_

#include <algorithm>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include <Eigen/Core>

#include <Graph.hpp>

#include <voro++/voro++.hh>

#include "CellList.h"
#include "Neighborhood.h"

namespace crayon
{

std::vector<Graph> buildGraphs(const std::vector<std::vector<int>> NL, unsigned int n_shells);

// std::vector<std::vector<int>>
std::tuple< std::vector<std::vector<int>>, std::vector<std::vector<double>> >
    VoroNeighbors(const Eigen::MatrixXf &R, const Eigen::VectorXf &L,
    const bool x_pbc, const bool y_pbc, const bool z_pbc);

std::vector<Eigen::VectorXi>
    CellNeighbors(const Eigen::MatrixXf &R, const Eigen::VectorXf &L,
        const bool x_pbc, const bool y_pbc, const bool z_pbc,
        const float rcut);

void export_VoroNeighbors(pybind11::module& m);

} // end namespace crayon

#endif // SRC_NEIGHBORS_H_
