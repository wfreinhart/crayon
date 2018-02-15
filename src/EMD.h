//
// EMD.h
// wraps Gary Doran's EMD code, which is based on Yossi Rubner's implementation
// https://github.com/garydoranjr/pyemd
//
// Copyright (c) 2018 Wesley Reinhart.
// This file is part of the crayon project, released under the Modified BSD License.

#ifndef SRC_EMD_H_
#define SRC_EMD_H_

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include <Eigen/Core>

#include <c_emd/emd.h>

#include <fastEMD/emd_hat.hpp>

namespace crayon
{

std::vector<std::vector<double>> EMDdists(const Eigen::MatrixXi &P, const Eigen::MatrixXi &Q);

double EMD(const Eigen::MatrixXi &P, const Eigen::MatrixXi &Q);

double fastEMD(const Eigen::VectorXd &P, const Eigen::VectorXd &Q);

double fastEMD(const std::vector<std::vector<double>> dists);

void export_EMD(pybind11::module& m);

} // end namespace crayon

#endif // SRC_EMD_H_
