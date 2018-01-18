import numpy as np

import sys
sys.path.insert(0,'/home/wfr/crayon/build')
sys.path.insert(0,'/home/wfr/crayon/test')
import crayon

from test_graphs import *
from test_utils import *

import unittest

class ValidatePyGraph(unittest.TestCase):
    def testSolo(self):
        for i in range(len(AList)):
            A = AList[i]
            idx = i + 1
            # test instantiation
            a = crayon.graph(A)
            A = a.adj()
            np.testing.assert_array_equal(AList[i],A)
            # test gdv calculation
            gdv = np.loadtxt('/home/wfr/crayon/test/gdv/%d.txt'%idx)
            A_gdv = a.gdv()
            np.testing.assert_array_equal(gdv,A_gdv)
            # test gdd calculation
            gdd = load_uneven('/home/wfr/crayon/test/gdd/%d.txt'%idx)
            A_gdd = a.gdd()
            np.testing.assert_array_equal(gdd,A_gdd)

if __name__ == "__main__":
    unittest.main()
