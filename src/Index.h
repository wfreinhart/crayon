//
// Index.h
// handles indexing for the cell list
//
// Copyright (c) 2018 Wesley Reinhart.
// This file is part of the crayon project, released under the Modified BSD License.

#ifndef SRC_INDEX_H_
#define SRC_INDEX_H_

#include <Eigen/Core>

namespace crayon
{

class Index2D
    {
    public:
    Index2D() : num_rows_(0), num_cols_(0) {};

    Index2D(unsigned int num_rows, unsigned int num_cols)
        : num_rows_(num_rows), num_cols_(num_cols) {};

    Index2D(Eigen::Vector2i dims)
        : num_rows_(dims(0)), num_cols_(dims(1)) {};

    unsigned int operator()(unsigned int i, unsigned int j) const
        {
        return j + i * num_cols_;
        }

    unsigned int operator()(Eigen::Vector2i ids) const
        {
        return ids(1) + ids(0) * num_cols_;
        }

    Eigen::Vector2i operator()(unsigned int idx) const
        {
        Eigen::Vector2i tuple;
        tuple(0) = idx / num_cols_;
        tuple(1) = idx - tuple(0) * num_cols_;
        return tuple;
        }

    Eigen::Vector2i shape()
        {
        Eigen::Vector2i tuple;
        tuple << num_rows_, num_cols_;
        return tuple;
        }

    private:
    unsigned int num_rows_;
    unsigned int num_cols_;
    };

class Index3D
    {
    public:
        Index3D() : w_(0), h_(0), d_(0) {}

        Index3D(unsigned int w, unsigned int h, unsigned int d)
            : w_(w), h_(h), d_(d) {}

        Index3D(Eigen::Vector3i dims)
            : w_(dims(0)), h_(dims(1)), d_(dims(2)) {}

        unsigned int operator()(unsigned int i, unsigned int j, unsigned int k) const
            {
            return (i * h_ + j) * d_ + k;
            }

        unsigned int operator()(Eigen::Vector3i ids) const
            {
            return (ids(0) * h_ + ids(1)) * d_ + ids(2);
            }

        Eigen::Vector3i operator()(unsigned int idx) const
            {
            Eigen::Vector3i tuple;
            tuple(0) = idx / (h_ * d_);
            tuple(1) = (idx % (h_ * d_)) / d_;
            tuple(2) = idx - (tuple(0) * h_ + tuple(1)) * d_;
            return tuple;
            }

        Eigen::Vector3i shape()
            {
            Eigen::Vector3i tuple;
            tuple << w_, h_, d_;
            return tuple;
            }

    private:
        unsigned int w_;
        unsigned int h_;
        unsigned int d_;
    };

} // end namespace crayon

#endif // SRC_INDEX_H_
