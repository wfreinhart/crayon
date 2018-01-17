#ifndef SRC_PY_GRAPH_H_
#define SRC_PY_GRAPH_H_

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include <Eigen/Core>

#include <graph/VertexVisitor.hpp>
#include <graph/EdgeVisitor.hpp>

#include <orca/Orca.hpp>
#include <Graph.hpp>

#include <libgraphlet/GDD.hpp>

#define GRAPHLET_SIZE 5

namespace crayon
    {

    class PyGraph
        {
        public:
        PyGraph(const Eigen::MatrixXi &A); // constructor
        ~PyGraph(); // destructor

        void build();

        const Eigen::MatrixXi getAdj() const { return A_; }

        Eigen::MatrixXi getGDV();
        void computeGDV();

        Eigen::MatrixXi getGDD();
        void computeGDD();

        private:
        // core data structures
        Eigen::MatrixXi A_;
        Graph G_;
        std::unique_ptr<orca::Orca> O_;
        // orca results
        Eigen::MatrixXi GDV_;
        bool computed_gdv_ = false;
        Eigen::MatrixXi GDD_;
        bool computed_gdd_ = false;
        };

    void export_PyGraph(pybind11::module& m);

    } // end namespace crayon

#endif // SRC_PY_GRAPH_H_
