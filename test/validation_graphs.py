#
# validation_graphs.py
# provides sample graphs for validation testing
#
# Copyright (c) 2018 Wesley Reinhart.
# This file is part of the crayon project, released under the Modified BSD License.

import numpy as np

o = np.array([1, 2, 2, 2, 3, 4, 3, 3, 4, 3,
              4, 4, 4, 4, 3, 4, 6, 5, 4, 5,
              6, 6, 4, 4, 4, 5, 7, 4, 6, 6,
              7, 4, 6, 6, 6, 5, 6, 7, 7, 5,
              7, 6, 7, 6, 5, 5, 6, 8, 7, 6,
              6, 8, 6, 9, 5, 6, 4, 6, 6, 7,
              8, 6, 6, 8, 7, 6, 7, 7, 8, 5,
              6, 6, 4],dtype=np.float)

w = 1. - np.log(o) / np.log(73.)

AList = []

AList.append( np.array([[1,1,1,0,0,0,0],
                        [1,1,1,1,0,0,0],
                        [1,1,1,0,1,0,0],
                        [0,1,0,1,0,1,0],
                        [0,0,1,0,1,0,1],
                        [0,0,0,1,0,1,1],
                        [0,0,0,0,1,1,1]]) )

AList.append( np.array([[1,1,1,1,0,0,0,0],
                        [1,1,0,1,1,0,0,0],
                        [1,0,1,0,0,1,0,0],
                        [1,1,0,1,0,1,1,0],
                        [0,1,0,0,1,0,1,0],
                        [0,0,1,1,0,1,1,1],
                        [0,0,0,1,1,1,1,1],
                        [0,0,0,0,0,1,1,1]]) )

AList.append( np.array([[1,1,1,1,0,0,0,0,0],
                        [1,1,0,0,1,1,0,0,0],
                        [1,0,1,1,1,0,1,0,0],
                        [1,0,1,1,0,0,1,0,0],
                        [0,1,1,0,1,1,0,1,0],
                        [0,1,0,0,1,1,0,1,0],
                        [0,0,1,1,0,0,1,1,1],
                        [0,0,0,0,1,1,1,1,1],
                        [0,0,0,0,0,0,1,1,1]]) )

AList.append( np.array([[1,1,1,0,1,0,1,0,0,0,0],
                        [1,1,0,0,1,0,1,0,0,0,0],
                        [1,0,1,1,0,1,0,1,0,0,0],
                        [0,0,1,1,1,0,0,1,1,0,0],
                        [1,1,0,1,1,0,0,0,1,0,0],
                        [0,0,1,0,0,1,1,1,0,1,0],
                        [1,1,0,0,0,1,1,0,0,1,0],
                        [0,0,1,1,0,1,0,1,0,0,1],
                        [0,0,0,1,1,0,0,0,1,0,1],
                        [0,0,0,0,0,1,1,0,0,1,1],
                        [0,0,0,0,0,0,0,1,1,1,1]]) )

AList.append( np.array([[1,1,1,1,1,0,0,0,0,0,0,0],
                        [1,1,0,0,1,1,1,0,0,0,0,0],
                        [1,0,1,1,0,0,0,1,1,0,0,0],
                        [1,0,1,1,0,1,0,0,1,0,0,0],
                        [1,1,0,0,1,0,0,1,0,1,0,0],
                        [0,1,0,1,0,1,1,0,0,0,1,0],
                        [0,1,0,0,0,1,1,0,0,1,1,0],
                        [0,0,1,0,1,0,0,1,0,1,0,1],
                        [0,0,1,1,0,0,0,0,1,0,1,1],
                        [0,0,0,0,1,0,1,1,0,1,0,1],
                        [0,0,0,0,0,1,1,0,1,0,1,1],
                        [0,0,0,0,0,0,0,1,1,1,1,1]]) )

AList.append( np.array([[1,1,0,1,0,0,0,0],
                        [1,1,1,1,0,1,0,0],
                        [0,1,1,0,0,1,0,0],
                        [1,1,0,1,1,0,1,0],
                        [0,0,0,1,1,0,1,0],
                        [0,1,1,0,0,1,1,1],
                        [0,0,0,1,1,1,1,1],
                        [0,0,0,0,0,1,1,1]]) )

AList.append( np.array([[1,1,0,1,1,0,0,0,0],
                        [1,1,1,0,1,1,0,0,0],
                        [0,1,1,1,0,1,1,0,0],
                        [1,0,1,1,0,0,1,0,0],
                        [1,1,0,0,1,0,0,1,0],
                        [0,1,1,0,0,1,0,1,1],
                        [0,0,1,1,0,0,1,0,1],
                        [0,0,0,0,1,1,0,1,1],
                        [0,0,0,0,0,1,1,1,1]]) )

AList.append( np.array([[1,1,1,1,0,0,1,0,0,0,0],
                        [1,1,0,0,1,0,1,0,0,0,0],
                        [1,0,1,1,0,1,0,1,0,0,0],
                        [1,0,1,1,1,0,0,0,1,0,0],
                        [0,1,0,1,1,0,0,0,1,0,0],
                        [0,0,1,0,0,1,1,1,0,1,0],
                        [1,1,0,0,0,1,1,0,0,1,0],
                        [0,0,1,0,0,1,0,1,1,0,1],
                        [0,0,0,1,1,0,0,1,1,0,1],
                        [0,0,0,0,0,1,1,0,0,1,1],
                        [0,0,0,0,0,0,0,1,1,1,1]]) )

AList.append( np.array([[1,1,1,0,1,1,0,0,0,0,0,0],
                        [1,1,1,1,0,0,0,1,0,0,0,0],
                        [1,1,1,0,0,0,1,0,1,0,0,0],
                        [0,1,0,1,1,0,0,1,0,1,0,0],
                        [1,0,0,1,1,1,0,0,0,1,0,0],
                        [1,0,0,0,1,1,1,0,0,0,1,0],
                        [0,0,1,0,0,1,1,0,1,0,1,0],
                        [0,1,0,1,0,0,0,1,1,0,0,1],
                        [0,0,1,0,0,0,1,1,1,0,0,1],
                        [0,0,0,1,1,0,0,0,0,1,1,1],
                        [0,0,0,0,0,1,1,0,0,1,1,1],
                        [0,0,0,0,0,0,0,1,1,1,1,1]]) )
