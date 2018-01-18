import numpy as np

import sys
sys.path.insert(0,'/home/wfr/crayon/build')
sys.path.insert(0,'/home/wfr/crayon/test')
import crayon

from test_graphs import *
from test_utils import *

import unittest

class ValidateComparison(unittest.TestCase):
    def testPair(self):
        # test pairwise gdv similarity and gdd agreement
        for i in range(len(AList)):
            A = AList[i]
            idx = i + 1
            a = crayon.graph(A)
            A = a.adj()
            A_gdv = a.gdv()
            A_gdd = a.gdd()
            for j in range(len(AList)):
                B = AList[j]
                jdx = j + 1
                b = crayon.graph(B)
                B_gdv = b.gdv()
                B_gdd = b.gdd()
                # test gdv similarity
                n = len(A) * len(B)
                gdvs = np.loadtxt('/home/wfr/crayon/test/gdvs/gdvs-%d-%d.txt'%(idx,jdx))
                V = np.zeros((len(A),len(B)))
                for k in range(len(gdvs)):
                    V[gdvs[k,0]-1,gdvs[k,1]-1] = gdvs[k,2]
                S = np.zeros((len(A),len(B)))
                # compare each node in the graph
                for l in range(len(A)):
                    for m in range(len(B)):
                        D = w * np.abs( np.log(A_gdv[l]+1.) - np.log(B_gdv[m]+1.) ) \
                            / np.log(np.amax(np.vstack((A_gdv[l],B_gdv[m])),0)+2.)
                        S[l,m] = 1. - np.sum( D ) / np.sum( w )
                np.testing.assert_array_almost_equal(np.sort(V.flatten()),np.sort(S.flatten()),3)
                # test comparitor
                T = crayon.gdvs(a,b)
                np.testing.assert_array_almost_equal(S,T,6)
                # test gdd agreement
                gdda = np.loadtxt('/home/wfr/crayon/test/gdda/gdda-%d-%d.txt'%(idx,jdx))
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
                np.testing.assert_array_almost_equal(gdda,Aj,3)
                # test comparitor
                D = crayon.gdda(a,b)
                np.testing.assert_array_almost_equal(Aj,D,6)

if __name__ == "__main__":
    unittest.main()
