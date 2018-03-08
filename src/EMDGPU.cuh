//
// EMDGPU.cuh
// CUDA implementation of Gary Doran's EMD code
//
// Copyright (c) 2018 Wesley Reinhart.
// This file is part of the crayon project, released under the Modified BSD License.

#ifndef SRC_EMD_GPU_CUH_
#define SRC_EMD_GPU_CUH_

#include <iostream>
#include <assert.h>
#include <cuda.h>

namespace crayon
{
namespace gpu
{
#define EPSILON 1e-12
#define FALSE 0
#define TRUE 1

enum COLOR {
    WHITE = 0,
    GRAY,
    BLACK
};

struct basic_variable {
    int row;
    int col;
    int idx;
    double flow;
    struct adj_node *adjacency;
    struct adj_node *current_adj;
    struct basic_variable *back_ptr;
    enum COLOR color;
};

struct adj_node {
    struct basic_variable *variable;
    struct adj_node *next;
};

void add(int N);

double pyemd(int n_x, double *weight_x,
    int n_y, double *weight_y,
    double **cost, double **flows);

} // end namespace gpu
} // end namespace crayon

#endif // SRC_EMD_GPU_CUH_