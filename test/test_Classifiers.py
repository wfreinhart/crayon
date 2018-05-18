#
# test_Classifiers.py
# unit tests for the crayon.classifiers python module
#
# Copyright (c) 2018 Wesley Reinhart.
# This file is part of the crayon project, released under the Modified BSD License.

import numpy as np

import os
test_path = os.path.abspath(os.path.dirname(__file__))
build_path = os.getcwd()
src_path = test_path[:test_path.rfind('/test')] + '/src'

import sys
sys.path.insert(0,build_path)
sys.path.insert(0,src_path)
import crayon

sys.path.insert(0,test_path)

from validation_graphs import *

import unittest

class TestCrayon(unittest.TestCase):
    # run this every time
    def setUp(self):
        pass

    def testClassifier(self):
        # default initialization for abstract class
        c = crayon.classifiers.Classifier()
        pass

    def testGraph(self):
        # default initialization
        A = np.zeros((5,5))
        x = crayon.classifiers.Graph(A)
        # with optional graphlet size
        x = crayon.classifiers.Graph(A,k=5)
        # check for null result
        np.testing.assert_array_equal(x.ngdv,np.zeros(73))
        # check for simple result
        B = np.array([[0,1,0],[1,0,0],[0,0,0]])
        y = crayon.classifiers.Graph(B,k=5)
        expected = np.zeros(73)
        expected[0] = 1.
        np.testing.assert_array_equal(y.ngdv,expected)
        # check for more involved result
        C = np.ones((10,10))
        z = crayon.classifiers.Graph(C,k=5)
        expected = np.zeros(73)
        expected[0] = 0.03644954
        expected[3] = 0.1437732
        expected[14] = 0.33074587
        expected[72] = 0.48903139
        np.testing.assert_array_almost_equal(z.ngdv,expected,7)

    def testLibrary(self):
        # default initialization for abstract class
        lib = crayon.classifiers.Library()
        pass

    def testGraphLibrary(self):
        lib = crayon.classifiers.GraphLibrary()
        graphs = [crayon.classifiers.Graph(A) for A in AList]
        lib.build(AList)
        # test find
        self.assertEqual(lib.find(graphs[0]),0)
        n = len(graphs) - 1
        self.assertEqual(lib.find(graphs[n]),n)
        # test encounter with existing graph
        # test encounter with new graph
        # test collect with identical libraries
        # test collect with non-identical libraries

if __name__ == "__main__":
    unittest.main(argv = ['test.py', '-v'])
