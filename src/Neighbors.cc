//
// Neighbors.cc
// wraps the voro++ library for calculation of neighbor lists
//
// Copyright (c) 2018 Wesley Reinhart.
// This file is part of the crayon project, released under the Modified BSD License.

#include "Neighbors.h"

namespace crayon
{

std::vector<Graph> buildGraphs(const std::vector<std::vector<int>> NL, unsigned int n_shells)
    {
    std::vector<Graph> graphs;
    graphs.resize(NL.size());
    for( unsigned int i = 0; i < NL.size(); i++ )
        {
        std::vector<int> idx = NL[i];
        // loop over the specified number of shells
        if( n_shells > 1 )
            {
            std::set<int> s( idx.begin(), idx.end() );
            unsigned int n = 1;
            while( n < n_shells )
                {
                for( int j : idx )
                    {
                    for( int k : NL[j] )
                        {
                        s.insert(k);
                        }
                    }
                idx.assign( s.begin(), s.end() );
                n += 1;
                }
            }
        // build edge list from neighbor list
        std::map<int,Graph::vertex_descriptor> map;
        Graph::vertex_descriptor n = idx.size();
        Graph G = Graph(n);
        G[boost::graph_bundle].label = "AdjMat";
        // fill nodes first
        for( unsigned int j = 0; j < n; j++ )
            {
            map[idx[j]] = j;
            Graph::vertex_descriptor v = map[j];
            G[v].label = std::to_string(idx[j]+1);
            }
        // now loop through nodes and fill edges
        // assumes symmetry! (Orca.cpp:47 symmetrizes anyways)
        for( int j : idx )
            {
            for( int k : idx )
                {
                if( std::find(NL[j].begin(), NL[j].end(), k) != NL[j].end() )
                    {
                    Graph::vertex_descriptor u = map[j];
                    Graph::vertex_descriptor v = map[k];
                    auto e = add_edge(u, v, G);
                    G[e.first].label = std::to_string(j+1) + "-" + std::to_string(k+1);
                    }
                }
            }
        graphs[i] = G;
        }
    return graphs;
    }

std::tuple< std::vector<std::vector<int>>, std::vector<std::vector<double>> >
VoroNeighbors(const Eigen::MatrixXf &R, const Eigen::VectorXf &L,
    const bool x_pbc, const bool y_pbc, const bool z_pbc)
    {
    double x_lo = -0.5*L(0);
    double x_hi =  0.5*L(0);
    double y_lo = -0.5*L(1);
    double y_hi =  0.5*L(1);
    double z_lo = -0.5*L(2);
    double z_hi =  0.5*L(2);
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
    std::vector<std::vector<double>> area(R.rows());
    voro::voronoicell_neighbor c;
    // compute each Voronoi cell in the container
    voro::c_loop_all cl(con);
    if(cl.start()) do if(con.compute_cell(c,cl))
                          {
                          unsigned int id = cl.pid();
                          c.neighbors(nl[id]);
                          c.face_areas(area[id]);
                          }
        while (cl.inc());
    return std::make_tuple(nl,area);
    }

std::vector<Eigen::VectorXi>
CellNeighbors(const Eigen::MatrixXf &R, const Eigen::VectorXf &L,
    const bool x_pbc, const bool y_pbc, const bool z_pbc,
    const float rcut)
    {
    std::vector<Eigen::VectorXi> nl(R.rows());
    CellList cells = CellList(R,L,rcut);
    unsigned int n_cells = cells.getNumCells().prod();
    for( unsigned int i = 0; i < n_cells; ++i )
        {
        Eigen::VectorXi cell_particles = cells.getParticles(i);
        Eigen::VectorXi adjacent_particles = cells.getAdjacentParticles(i);
        for( unsigned int j = 0; j < cell_particles.size(); ++j )
            {
            unsigned int idx = cell_particles(j);
            nl[idx].resize(adjacent_particles.size());
            for( unsigned int k = 0; k < adjacent_particles.size(); k++ )
                {
                nl[idx](k) = adjacent_particles(k);
                }
            }
        }
    return nl;
    }

void export_VoroNeighbors(pybind11::module& m)
    {
    m.def("voropp",&VoroNeighbors);
    m.def("cellpp",&CellNeighbors);
    m.def("buildGraphs",&buildGraphs);
    }

}  // end namespace crayon
