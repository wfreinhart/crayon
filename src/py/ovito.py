#
# ovito.py
# Ovito modification to read NGA color maps from file
#
# Copyright (c) 2018 Wesley Reinhart.
# This file is part of the crayon project, released under the Modified BSD License.

import ovito
import numpy as np

def modify(frame, input, output):
	source = input.attributes['SourceFile']
	print(source)
	cmap = np.loadtxt(source + '.cmap')
	color_property = output.create_particle_property(ovito.data.ParticleProperty.Type.Color)
	N = len(color_property.marray)
	for i in range(N):
		color_property.marray[i] = tuple(cmap[i,1:])
