//
// EMDGPU.h
// wraps a CUDA implementation of Gary Doran's EMD code
//
// Copyright (c) 2018 Wesley Reinhart.
// This file is part of the crayon project, released under the Modified BSD License.

#ifndef SRC_EMD_GPU_H_
#define SRC_EMD_GPU_H_

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include <Eigen/Core>

#include "EMDGPU.cuh"

namespace crayon
{

double EMDGPU(const Eigen::MatrixXd &P, const Eigen::MatrixXd &Q);

void export_EMDGPU(pybind11::module& m);

} // end namespace crayon

#endif // SRC_EMD_GPU_H_
