#
# test_Comparison.py
# unit tests for the crayon Comparison C++ methods
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

class ValidateComparison(unittest.TestCase):
    # run this every time
    def setUp(self):
        self.aList = [_crayon.graph(A) for A in AList]

    # test pairwise GDV similarity validity
    def testValidateGDVS(self):
        for i in range(len(AList)):
            A_gdv = self.aList[i].gdv()
            for j in range(len(AList)):
                B_gdv = self.aList[j].gdv()
                # test gdv similarity
                n = len(AList[i]) * len(AList[j])
                gdvs = np.loadtxt('/home/wfr/crayon/test/gdvs/gdvs-%d-%d.txt'%(i+1,j+1))
                V = np.zeros((len(AList[i]),len(AList[j])))
                for k in range(len(gdvs)):
                    V[gdvs[k,0]-1,gdvs[k,1]-1] = gdvs[k,2]
                S = np.zeros((len(AList[i]),len(AList[j])))
                # compare each node in the graph
                for l in range(len(AList[i])):
                    for m in range(len(AList[j])):
                        D = w * np.abs( np.log(A_gdv[l]+1.) - np.log(B_gdv[m]+1.) ) \
                            / np.log(np.amax(np.vstack((A_gdv[l],B_gdv[m])),0)+2.)
                        S[l,m] = 1. - np.sum( D ) / np.sum( w )
                # test calculated values against libgraphlet
                # note: libgraphlet returns the nodes in the wrong order!
                #       we can only check that the set of values match...
                np.testing.assert_array_almost_equal(np.sort(V.flatten()),np.sort(S.flatten()),3)
                # test comparitor against calculated values
                T = _crayon.gdvs(self.aList[i],self.aList[j])
                np.testing.assert_array_almost_equal(S,T,6)

    # test pairwise GDD agreement validity
    def testValidateGDDA(self):
        for i in range(len(AList)):
            A_gdd = self.aList[i].gdd()
            for j in range(len(AList)):
                B_gdd = self.aList[j].gdd()
                # test gdd agreement
                gdda = np.loadtxt('/home/wfr/crayon/test/gdda/gdda-%d-%d.txt'%(i+1,j+1))
                n = np.max((A_gdd.shape[1],B_gdd.shape[1]))
                S_A = np.hstack((A_gdd,np.zeros((A_gdd.shape[0],n-A_gdd.shape[1]))))
                N_A = np.zeros(S_A.shape)
                for k in range(1,S_A.shape[1]):
                    S_A[:,k] = S_A[:,k] / k
                T_A = np.sum(S_A,axis=1)
                for k in range(S_A.shape[0]):
                    if T_A[k] == 0:
                        continue
                    N_A[k,:] = S_A[k,:] / T_A[k]
                S_B = np.hstack((B_gdd,np.zeros((B_gdd.shape[0],n-B_gdd.shape[1]))))
                N_B = np.zeros(S_B.shape)
                for k in range(1,S_B.shape[1]):
                    S_B[:,k] = S_B[:,k] / k
                T_B = np.sum(S_B,axis=1)
                for k in range(S_B.shape[0]):
                    if T_B[k] == 0:
                        continue
                    N_B[k,:] = S_B[k,:] / T_B[k]
                Aj = 1. - 1./np.sqrt(2)*np.sum((N_A-N_B)**2.,1)**0.5
                # test calculated values against libgraphlet
                np.testing.assert_array_almost_equal(gdda,Aj,3)
                # test comparitor against calculated values
                D = _crayon.gdda(self.aList[i],self.aList[j])
                np.testing.assert_array_almost_equal(Aj,D,6)

if __name__ == "__main__":
    unittest.main(argv = ['test.py', '-v'])
