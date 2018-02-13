#
# __init__.py
# defines crayon module and loads submodules
#
# Copyright (c) 2018 Wesley Reinhart.
# This file is part of the crayon project, released under the Modified BSD License.

from crayon import _crayon
import color
import nga
import neighborlist
import io
try:
    import dmap
except:
    print('Warning: diffusion map features disabled')
try:
    import parallel
except:
    print('Warning: parallel features disabled')
