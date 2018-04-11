//
// Neighborhood.h
// wraps the libgraphlet/Orca calculation of GDVs and GDDs
//
// Copyright (c) 2018 Wesley Reinhart.
// This file is part of the crayon project, released under the Modified BSD License.

#ifndef SRC_NEIGHBORHOOD_H_
#define SRC_NEIGHBORHOOD_H_

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include <Eigen/Core>

#include <graph/VertexVisitor.hpp>
#include <graph/EdgeVisitor.hpp>

#include <orca/Orca.hpp>
#include <Graph.hpp>

#include <libgraphlet/GDD.hpp>

namespace crayon
    {

    class Neighborhood
        {
        public:
        Neighborhood(); // empty constructor
        Neighborhood(const Eigen::MatrixXi &A); // construct from numpy adjacency matrix
        Neighborhood(const Eigen::MatrixXi &A, const int k);
        Neighborhood(const Graph &G); // construct from Graph object
        Neighborhood(const Graph &G, const int k);
        ~Neighborhood(); // destructor

        void setGraph(const Graph &G) { G_ = G; }

        void buildFromAdj();
        void setup();

        const Eigen::MatrixXi getAdj() const { return A_; }
        const int getK() const { return k_; }

        Eigen::MatrixXi getGDV();
        void computeGDV();

        Eigen::MatrixXi getGDD();
        void computeGDD();

        private:
        // core data structures
        Eigen::MatrixXi A_;
        Graph G_;
        int k_ = 5;
        std::unique_ptr<orca::Orca> O_;
        // orca results
        Eigen::MatrixXi GDV_;
        bool computed_gdv_ = false;
        Eigen::MatrixXi GDD_;
        bool computed_gdd_ = false;
        };

    void export_Neighborhood(pybind11::module& m);

    } // end namespace crayon

#endif // SRC_PY_GRAPH_H_
