#
# test_io.py
# unit tests for the crayon.io python module
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

class TestIO(unittest.TestCase):
    # run this every time
    def setUp(self):
        pass

    def testReadXYZ(self):
        xyz, L = crayon.io.readXYZ(test_path+'/xyz/test2D.xyz')
        expected = np.zeros((9,3))
        k = 0
        for i in range(3):
            for j in range(3):
                expected[k,:] = (j,i,0.)
                k += 1
        np.testing.assert_array_almost_equal(expected,xyz)
        np.testing.assert_array_almost_equal(np.ones(3)*10.,L)

if __name__ == "__main__":
    unittest.main(argv = ['test.py', '-v'])
