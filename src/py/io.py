#
# io.py
# functions for reading and writing files
#
# Copyright (c) 2018 Wesley Reinhart.
# This file is part of the crayon project, released under the Modified BSD License.

from __future__ import print_function
import sys

from crayon import parallel

import numpy as np
import xml.etree.ElementTree as etree

try:
    import gsd
    from gsd import fl
    from gsd import hoomd
    foundGSD = True
except:
    foundGSD = False

def readXML(snap,reader_input):
    filename = reader_input
    # read values from file
    config  = open(filename,'rb+')
    tree = etree.parse(config)
    root = tree.getroot()
    elem = root.getiterator("box")[0]
    lx = float(elem.attrib["lx"])
    ly = float(elem.attrib["ly"])
    lz = float(elem.attrib["lz"])
    L = np.array([lx,ly,lz])
    elem = root.getiterator("position")[0]
    txt = elem.text
    dat = np.fromstring(txt,sep=' ')
    R = np.reshape(dat,(-1,3))
    config.close()
    # assign values to Snapshot
    snap.N = len(R)
    snap.xyz = R
    snap.L = L

def readGSD(snap,reader_input):
    if not foundGSD:
        raise RuntimeError('GSD module not found')
    filename = reader_input
    # read trajectory from gsd file
    gsd_file = gsd.fl.GSDFile(filename,'rb')
    gsd_traj = gsd.hoomd.HOOMDTrajectory(gsd_file)
    gsd_frame = gsd_traj[-1]
    # read values from file
    L = gsd_frame.configuration.box[:3]
    R = gsd_frame.particles.position[:,:3]
    # assign values to Snapshot
    snap.N = len(R)
    snap.xyz = R
    snap.L = L

def readListParallel(filename):
    p = parallel.ParallelTask()
    comm, size, rank, master = parallel.info()
    filenames = None
    if master:
        with open(filename,'r') as fid:
            lines = fid.readlines()
        filenames = [x.strip() for x in lines]
        filenames = [x for x in filenames if x[0] != '#']
    filenames = p.shareData(filenames)
    return filenames
