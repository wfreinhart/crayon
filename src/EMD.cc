//
// EMD.cc
// wraps Gary Doran's EMD code, which is based on Yossi Rubner's implementation
// https://github.com/garydoranjr/pyemd
//
// Copyright (c) 2018 Wesley Reinhart.
// This file is part of the crayon project, released under the Modified BSD License.

#include "EMD.h"

namespace crayon
{

double EMD(const Eigen::MatrixXd &P, const Eigen::MatrixXd &Q)
    {
    int n_x = P.rows();
    int n_y = Q.rows();
    double *weight_x = (double *) malloc(n_x * sizeof(double));
    for( int i = 0; i < n_x; i++ ) { weight_x[i] = 1./n_x; }
    double *weight_y = (double *) malloc(n_y * sizeof(double));
    for( int j = 0; j < n_y; j++ ) { weight_y[j] = 1./n_y; }
    double **cost = (double **) malloc(n_x * sizeof(double *));
    for( int i = 0; i < n_x; i++ )
        {
        cost[i] = new double[n_y];
        for( int j = 0; j < n_y; j++ )
            {
            // use Euclidean distance between 73-dimensional graphlet histograms
            Eigen::VectorXd delta = P.row(i) - Q.row(j);
            cost[i][j] = sqrt( delta.dot(delta) );
            }
        }
    double **flows_data_ptr = NULL;
    double d = emd(n_x, weight_x, n_y, weight_y, cost, flows_data_ptr);
    return d;
    }

void export_EMD(pybind11::module& m)
    {
    m.def("emd",&EMD);
    }

}  // end namespace crayon
