//
// Neighbors.cc
// wraps the voro++ library for calculation of neighbor lists
//
// Copyright (c) 2018 Wesley Reinhart.
// This file is part of the crayon project, released under the Modified BSD License.

#include "Neighbors.h"

namespace crayon
{

std::vector<std::vector<int>> VoroNeighbors(const Eigen::MatrixXf &R, const Eigen::VectorXf &L)
    {
    double x_lo = -0.5*L(0);
    double x_hi =  0.5*L(0);
    double y_lo = -0.5*L(1);
    double y_hi =  0.5*L(1);
    double z_lo = -0.5*L(2);
    double z_hi =  0.5*L(2);
    bool x_pbc = true;
    bool y_pbc = true;
    bool z_pbc = true;
    voro::pre_container precon(x_lo,x_hi,y_lo,y_hi,z_lo,z_hi,x_pbc,y_pbc,z_pbc);
    int nx = 0;
    int ny = 0;
    int nz = 0;
    int init_mem = 8;
    // fill in particle data
    for( int i = 0; i < R.rows(); i++ )
        {
        precon.put(i,R(i,0),R(i,1),R(i,2));
        }
    precon.guess_optimal(nx,ny,nz);
    voro::container con(x_lo,x_hi,y_lo,y_hi,z_lo,z_hi,nx,ny,nz,x_pbc,y_pbc,z_pbc,init_mem);
    precon.setup(con);
    //
    std::vector<std::vector<int>> nl(R.rows());
    voro::voronoicell_neighbor c;
    // compute each Voronoi cell in the container
    voro::c_loop_all cl(con);
    if(cl.start()) do if(con.compute_cell(c,cl))
                          {
                          unsigned int id = cl.pid();
                          c.neighbors(nl[id]);
                          }
        while (cl.inc());
    return nl;
    }

void export_VoroNeighbors(pybind11::module& m)
    {
    m.def("voropp",&VoroNeighbors);
    }

}  // end namespace crayon
