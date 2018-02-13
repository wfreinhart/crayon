#
# test_Crayon.py
# unit tests for the crayon python module
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
import py as crayon

sys.path.insert(0,test_path)

import unittest

class TestCrayon(unittest.TestCase):
    # run this every time
    def setUp(self):
        pass

    def testInit(self):
        pass

if __name__ == "__main__":
    unittest.main(argv = ['test.py', '-v'])
