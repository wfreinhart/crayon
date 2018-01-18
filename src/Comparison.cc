#include "Comparison.h"

namespace crayon
{

// from libgraphlet/Similarity.cpp
const double AFFECTED[73] = {
		1., 2., 2., 2., 3., 4., 3., 3., 4., 3.,
		4., 4., 4., 4., 3., 4., 6., 5., 4., 5.,
		6., 6., 4., 4., 4., 5., 7., 4., 6., 6.,
		7., 4., 6., 6., 6., 5., 6., 7., 7., 5.,
		7., 6., 7., 6., 5., 5., 6., 8., 7., 6.,
		6., 8., 6., 9., 5., 6., 4., 6., 6., 7.,
		8., 6., 6., 8., 7., 6., 7., 7., 8., 5.,
		6., 6., 4.};

Eigen::MatrixXd GDVSimilarity(PyGraph &A, PyGraph &B)
    {
    // generate weights for GDV comparison
    // from libgraphlet/Similarity.cpp
    const unsigned int n = orca::ORBITS[GRAPHLET_SIZE];
    std::vector<double> w(n);
    for(unsigned int k = 0; k < n; ++k)
        {
        w[k] = 1. - log(AFFECTED[k]) / log(n);
        }
    double wS = std::accumulate(w.begin(), w.end(), 0.);
    // fetch the GDVs from the graph objects
    Eigen::MatrixXi A_gdv = A.getGDV();
    Eigen::MatrixXi B_gdv = B.getGDV();
    // allocate memory
    Eigen::MatrixXd S = Eigen::MatrixXd::Zero(A_gdv.rows(),B_gdv.rows());
    // compare nodes pairwise
    for( unsigned int i = 0; i < A_gdv.rows(); i++ )
        {
        for( unsigned int j = 0; j < B_gdv.rows(); j++ )
            {
            double D = 0;
            for( unsigned int k = 0; k < n; k++ )
                {
                double A_ik = A_gdv(i,k);
                double B_jk = B_gdv(j,k);
                double num = fabs( log(A_ik+1.) - log(B_jk+1.) );
                double denom = log( std::max(A_ik,B_jk) + 2. );
                D += w[k] * num / denom;
                }
            S(i,j) = 1. - D / wS;
            }
        }
    return S;
    }

Eigen::VectorXd GDDAgreement(PyGraph &A, PyGraph &B)
    {
    const unsigned int n = orca::ORBITS[GRAPHLET_SIZE];
    // fetch the GDDs from the graph objects
    Eigen::MatrixXi A_gdd = A.getGDD();
    Eigen::MatrixXi B_gdd = B.getGDD();
    // allocate memory
    unsigned int w = std::max(A_gdd.cols(),B_gdd.cols());
    Eigen::MatrixXd N_A = Eigen::MatrixXd::Zero(n,w);
    Eigen::MatrixXd N_B = Eigen::MatrixXd::Zero(n,w);
    // normalize by degree and graphlet area
    for( unsigned int i = 0; i < n; i++ )
        {
        double aS = 0.;
        double bS = 0.;
        for( unsigned int j = 1; j < w; j++ )
            {
            if( j < A_gdd.cols() )
                {
                double a = double(A_gdd(i,j)) / double(j);
                N_A(i,j) = a;
                aS += a;
                }
            if( j < B_gdd.cols() )
                {
                double b = double(B_gdd(i,j)) / double(j);
                N_B(i,j) = b;
                bS += b;
                }
            }
        if( aS > 0. ) N_A.row(i) /= aS;
        if( bS > 0. ) N_B.row(i) /= bS;
        }
    Eigen::VectorXd Aj(n);
    for( unsigned int k = 0; k < n; k++ )
        {
        double delta = 0.;
        for( unsigned int j = 0; j < w; j++ )
            {
            double d = ( N_A(k,j) - N_B(k,j) );
            delta += d * d;
            }
        Aj(k) = 1. - 1. / sqrt(2.) * sqrt(delta);
        }
    return Aj;
    }

void export_Comparison(pybind11::module& m)
    {
    m.def("gdvs",&GDVSimilarity);
    m.def("gdda",&GDDAgreement);
    }

}  // end namespace crayon
