//
// CellList.h
// creates a cell list from particle positions
//
// Copyright (c) 2018 Wesley Reinhart.
// This file is part of the crayon project, released under the Modified BSD License.

#ifndef SRC_CELL_LIST_H_
#define SRC_CELL_LIST_H_

#include <math.h>

#include <pybind11/stl.h>

#include <Eigen/Core>

#include "Index.h"

namespace crayon
{

class CellList
    {
    public:
    CellList(const Eigen::MatrixXf &R, const Eigen::VectorXf &L, const float rcut);
    ~CellList();

    void build();

    Eigen::Vector3f wrap(Eigen::Vector3f xyz)
    {
    for( int i = 0; i < 3; ++i )
        {
        if( std::abs(xyz(i)) > 0.5 * L_(i) )
            {
            xyz(i) -= L_(i) * std::round( xyz(i) / L_(i) );
            }
        }
    return xyz;
    }

    Eigen::Vector3i computeCells(Eigen::Vector3f xyz)
        {
        Eigen::Vector3f xyz_wrapped = wrap(xyz) + 0.5 * L_;
        Eigen::Vector3f frac_bin = xyz_wrapped.array() / cell_width_.array();
        Eigen::Vector3i cell_coords;
        for( int i = 0; i < 3; ++i )
            {
            cell_coords(i) = static_cast<int>(frac_bin(i));
            }
        return cell_coords;
        }

    Eigen::VectorXi getParticles(unsigned int cell)
        {
        Eigen::VectorXi particles(cell_size_(cell));
        unsigned int h = cell_head_(cell);
        for( int i = 0; i < cell_size_(cell); ++i )
            {
            particles(i) = cell_list_(h+i);
            }
        return particles;
        }

    Eigen::VectorXi getCellNeighbors(unsigned int cell)
        {
        Eigen::VectorXi neighbors(27);
        Eigen::Vector3i cell_3d = cell_indexer_(cell);
        int c = 0;
        Eigen::Vector3i coords;
        for( int i = -1; i < 2; ++i )
            {
            coords(0) = cell_3d(0) + i;
            for( int j = -1; j < 2; ++j )
                {
                coords(1) = cell_3d(1) + j;
                for( int k = -1; k < 2; ++k )
                    {
                    coords(2) = cell_3d(2) + k;
                    neighbors(c) = cell_indexer_(wrapCell3DIndex(coords));
                    c += 1;
                    }
                }
            }
        return neighbors;
        }

    Eigen::VectorXi getAdjacentParticles(unsigned int cell)
	{
	// get the cells adjacent to that one
	Eigen::VectorXi neighboring_cells = getCellNeighbors(cell);
	// loop through and compile particles
	unsigned int n_particles = 0;
	for( int i = 0; i < 27; ++i )
	    {
	    unsigned int this_cell = neighboring_cells(i);
	    n_particles += cell_size_(this_cell);
	    }
	// put particles into the list
	Eigen::VectorXi adjacent_particles(n_particles);
	int h = 0;
	for( int i = 0; i < 27; ++i )
	    {
	    unsigned int this_cell = neighboring_cells(i);
	    unsigned int this_cell_size = cell_size_(this_cell);
	    unsigned int this_cell_head = cell_head_(this_cell);
	    adjacent_particles.segment(h,this_cell_size) = cell_list_.segment(this_cell_head,this_cell_size);
	    h += this_cell_size;
	    }
	return adjacent_particles;
	}

    Eigen::Vector3i wrapCell3DIndex(Eigen::Vector3i coords)
        {
        Eigen::Vector3i dims = cell_indexer_.shape();
        Eigen::Vector3i wrapped_coords = coords;
        for( int i = 0; i < 3; ++i )
            {
            while( wrapped_coords(i) >= dims(i) )
                {
                wrapped_coords(i) -= dims(i);
                }
            while( wrapped_coords(i) < 0 )
                {
                wrapped_coords(i) += dims(i);
                }
            }
        return wrapped_coords;
        }

    private:
    Eigen::Vector3f L_;

    Eigen::Vector3f cell_width_;
    Eigen::Vector3i num_cells_;
    Eigen::VectorXi cell_list_;
    Eigen::VectorXi cell_size_;
    Eigen::VectorXi cell_head_;
    Eigen::VectorXi particle_cells_;
    Index3D cell_indexer_;
    };

} // end namespace crayon

#endif // SRC_CELL_LIST_H_
