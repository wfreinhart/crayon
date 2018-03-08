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

#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include <Eigen/Core>

namespace crayon
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

double EMD(const Eigen::MatrixXd &P, const Eigen::MatrixXd &Q);

double pyemd(int n_x, double *weight_x,
    int n_y, double *weight_y,
    double **cost, double **flows);

struct basic_variable **initialize_flow(int n_x, double *weight_x,
                                       int n_y, double *weight_y,
                                       double **cost);

struct basic_variable *init_basic(int row, int col, double flow);

void insert_basic(struct basic_variable **basis, int size,
                  struct basic_variable *node);

void remove_basic(struct basic_variable **basis, int size,
                  struct basic_variable *node);

void reset_current_adj(struct basic_variable **basis, int size);

void destruct_basis(struct basic_variable **basis, int size);

double *vector_malloc(int n);

double *vector_copy(double *v, int n);

void export_EMD(pybind11::module& m);

} // end namespace crayon

#endif // SRC_EMD_H_
