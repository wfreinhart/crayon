#
# __init__.py
# defines crayon module and loads submodules
#
# Copyright (c) 2018 Wesley Reinhart.
# This file is part of the crayon project, released under the Modified BSD License.

import _crayon
import classifiers
try:
    import dmap
except:
    print('Warning: diffusion map features disabled')
import io
import neighborlist
import nga
try:
    import parallel
except:
    print('Warning: parallel features disabled')
