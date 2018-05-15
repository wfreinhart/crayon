#
# test_PyGraph.py
# unit tests for the crayon::PyGraph C++ class
#
# Copyright (c) 2018 Wesley Reinhart.
# This file is part of the crayon project, released under the Modified BSD License.

import numpy as np

import os
test_path = os.path.abspath(os.path.dirname(__file__))
build_path = os.getcwd()

import sys
sys.path.insert(0,build_path)
import _crayon

sys.path.insert(0,test_path)
from validation_graphs import *
from utils import *

import unittest

class TestPyGraph(unittest.TestCase):
    # run this every time
    def setUp(self):
        self.aList = [_crayon.neighborhood(A) for A in AList]

    # test adjacency matrix assignment and retrieval
    def testAdj(self):
        for i in range(len(self.aList)):
            np.testing.assert_array_equal(AList[i],self.aList[i].adj())

    # test retrieval of GDV
    def testGDV(self):
        for i in range(len(self.aList)):
            gdv = self.aList[i].gdv()
            self.assertEqual(gdv.shape,(len(AList[i]),73))

    # test retrieval of GDD
    def testGDD(self):
        for i in range(len(self.aList)):
            gdd = self.aList[i].gdd()
            self.assertEqual(gdd.shape[0],73)

class ValidatePyGraph(unittest.TestCase):
    # run this every time
    def setUp(self):
        self.aList = [_crayon.neighborhood(A) for A in AList]

    # test GDV validity
    def testValidateGDV(self):
        for i in range(len(AList)):
            gdv = np.loadtxt(test_path + '/gdv/%d.txt'%(i+1))
            np.testing.assert_array_equal(gdv,self.aList[i].gdv())

    # test GDD validity
    def testValidateGDD(self):
        for i in range(len(AList)):
            gdd = load_uneven(test_path + '/gdd/%d.txt'%(i+1))
            np.testing.assert_array_equal(gdd,self.aList[i].gdd())

if __name__ == "__main__":
    unittest.main(argv = ['test.py', '-v'])
