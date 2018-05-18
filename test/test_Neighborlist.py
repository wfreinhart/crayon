#
# test_Neighborlist.py
# unit tests for the crayon.neighborlist python module
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

import unittest

class TestNeighborlist(unittest.TestCase):
    # run this every time
    def setUp(self):
        pass

    def testInit(self):
        pass

    def testCutoff(self):
        nl = crayon.neighborlist.Cutoff()
        nl.rcut = 1.44
        xyz = np.zeros((9,3))
        k = 0
        for i in range(3):
            for j in range(3):
                xyz[k,:] = (j,i,0.)
                k += 1
        box = np.ones(3)*10.
        snap = crayon.nga.Snapshot(xyz,box)
        nbr = nl.getNeighbors(snap)
        expected = [[0,1,3,4],
                    [0,1,2,3,4,5],
                    [1,2,4,5],
                    [0,1,3,4,6,7],
                    [0,1,2,3,4,5,6,7,8],
                    [1,2,4,5,7,8],
                    [3,4,6,7],
                    [3,4,5,6,7,8],
                    [4,5,7,8]]
        for i in range(xyz.shape[0]):
            np.testing.assert_array_equal(np.array(expected[i]),np.sort(np.array(nbr[i])))

if __name__ == "__main__":
    unittest.main(argv = ['test.py', '-v'])
