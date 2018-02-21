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

std::vector<std::vector<double>> EMDdists(const Eigen::MatrixXi &P, const Eigen::MatrixXi &Q)
    {
    std::vector<std::vector<double>> dists(P.rows(), std::vector<double>(Q.rows(), 0.));
    for( unsigned int i = 0; i < P.rows(); i++ )
        {
        Eigen::VectorXd phist = P.row(i).cast<double>();
        phist /= phist.sum();
        for( unsigned int j = 0; j < Q.rows(); j++ )
            {
            Eigen::VectorXd qhist = Q.row(j).cast<double>();
            qhist /= qhist.sum();
            dists[i][j] = fastEMD(phist,qhist);
            }
        }
    return dists;
    }

double EMD(const Eigen::MatrixXi &P, const Eigen::MatrixXi &Q)
    {
    std::vector<std::vector<double>> dists = EMDdists(P,Q);
    return fastEMD(dists);
    }

double fastEMD(const Eigen::VectorXd &P, const Eigen::VectorXd &Q)
    {
    std::vector<double> pvec(P.data(), P.data() + P.size());
    std::vector<double> qvec(Q.data(), Q.data() + Q.size());
    std::vector<std::vector<double>> dists(pvec.size(), std::vector<double>(qvec.size(), 0.));
    for( unsigned int i = 0; i < pvec.size(); i++ )
        {
        for( unsigned int j = 0; j < qvec.size(); j++ )
            {
            dists[i][j] = std::abs(double(j)-double(i));
            }
        }
    double d = emd_hat_gd_metric<double>()(pvec, qvec, dists);
    return d;
    }

// double fastEMD(const std::vector<std::vector<double>> D)
//     {
//     unsigned int n = std::max(D.size(),D[0].size());
//     std::vector<std::vector<double>> dists(D.size(), std::vector<double>(D[0].size(), 0.));
//     for( unsigned int i = 0; i < D.size(); i++ )
//         {
//         for( unsigned int j = 0; j < D[i].size(); j++ )
//             {
//             dists[i][j] = D[i][j];
//             }
//         }
//     std::vector<double> pvec(n, 0.);
//     for( unsigned int i = 0; i < D.size(); i ++ ) pvec[i] = 1. / D.size();
//     std::vector<double> qvec(n, 0.);
//     for( unsigned int j = 0; j < D[0].size(); j ++ ) qvec[j] = 1. / D[0].size();
//     return emd_hat_gd_metric<double>()(pvec, qvec, dists);
//     }

double fastEMD(const std::vector<std::vector<double>> dists)
    {
    std::vector<double> pvec(dists.size(), 1./dists.size());
    std::vector<double> qvec(dists[0].size(), 1./dists[0].size());
    unsigned int n = std::max(pvec.size(),qvec.size());
    std::vector<std::vector<double>> flows(n, std::vector<double>(n, 0.));
    double extra_mass_penalty = 0.;
    double d = emd_hat_gd_metric<double, WITHOUT_EXTRA_MASS_FLOW>()(pvec, qvec, dists, extra_mass_penalty, &flows);
    double r = 0.;
    double s = 0.;
    for( unsigned int i = 0; i < dists.size(); i++ )
        {
        for( unsigned int j = 0; j < dists[i].size(); j++ )
            {
            r += dists[i][j] * flows[i][j];
            s += flows[i][j];
            }
        }
    return r / s;
    }

void export_EMD(pybind11::module& m)
    {
    m.def("emd_dists",&EMDdists);
    m.def("emd",&EMD);
    }

}  // end namespace crayon
