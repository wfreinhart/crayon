#include "CellList.h"

namespace crayon
{

CellList::CellList(const Eigen::MatrixXf &R, const Eigen::VectorXf &L, const float rcut)
    {
    for( int i = 0; i < 3; ++i )
        {
        L_(i) = L(i);
        }
    float box_size = rcut * 2.0 / 3.0; // radius is 1.5 * box length
    for( int i = 0; i < 3; ++i )
        {
        num_cells_(i) = std::floor( L_(i) / box_size );
        }

    for( int i = 0; i < 3; ++i )
        {
        if( num_cells_(i) < 3 )
            {
            num_cells_(i) = 3;
            }
        }

    cell_indexer_ = Index3D(num_cells_);

    for( int i = 0; i < 3; ++i )
        {
        cell_width_(i) = L_(i) / static_cast<float>(num_cells_(i));
        }

    unsigned int N = R.rows();
    particle_cells_.resize(N);
    particle_cells_.setZero();
    cell_list_.resize(N);
    cell_list_.setZero();

    unsigned int Ncells = num_cells_.prod();
    cell_size_.resize(Ncells);
    cell_size_.setZero();
    cell_head_.resize(Ncells);
    cell_head_.setZero();

    // calculate the cell for each particle
    for( int i = 0; i < N; ++i )
        {
        Eigen::Vector3i cell_coord = computeCells(R.row(i));
        int cell = cell_indexer_( wrapCell3DIndex(cell_coord) );
        particle_cells_(i) = cell;
        cell_size_(cell) += 1;
        }

    // calculate cell heads from cell sizes
    int idx = 0;
    for( int i = 0; i < Ncells; ++i )
        {
        cell_head_(i) = idx;
        idx += cell_size_(i);
        }

    // populate cell list
    int h, c;
    for( int i = 0; i < Ncells; ++i )
        {
        h = cell_head_(i);
        c = 0;
        for( int j = 0; j < N; ++j )
            {
            if( particle_cells_(j) == i )
                {
                cell_list_(h+c) = j;
                c += 1;
                }
            }
        }
    }

CellList::~CellList()
    {
    }

} // end namespace crayon
