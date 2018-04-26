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

try:
    import xml.etree.ElementTree as etree
    foundETree = True
except:
    foundETree = False

try:
    import gsd
    from gsd import fl
    from gsd import hoomd
    foundGSD = True
except:
    foundGSD = False

def readXYZ(snap,reader_input):
    filename = reader_input
    # read values from file
    with open(filename,'r') as config:
        lines = config.readlines()
    N = int(lines[0])
    if 'Lattice=' in lines[1]:
        L = np.asarray([float(x) for x in lines[1].replace('Lattice=','').replace('"','').split()[::4]])
    elif len(lines[1].split) == 3:
        L = np.asarray([float(x) for x in lines[1].split()])
    else:
        raise RuntimeError('unexpected box format in file %s'%filename)
    R = np.zeros((N,3))
    T_list = np.zeros(N,dtype=str)
    for i, l in enumerate(lines[2:]):
        R[i,:] = [float(x) for x in l.split()[1:]]
        T_list[i] = l.split()[0]
    types = list(np.unique(T_list))
    T = np.zeros(N,dtype=np.int)
    for i, t in enumerate(types):
        T[T_list==t] = i
    # assign values to Snapshot
    snap.N = len(R)
    snap.xyz = R
    snap.L = L
    snap.T = T

def readXML(snap,reader_input):
    if not foundETree:
        raise RuntimeError('xml.etree.ElementTree module not found')
    filename = reader_input
    # read values from file
    config  = open(filename,'r')
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
    N = len(R)
    try:
        elem = root.getiterator("type")[0]
        txt = elem.text
        dat = txt.replace(' ','').split('\n')
        T_list = np.array([d for d in dat if len(d) > 0])
        types = list(np.unique(T_list))
        T = np.zeros(N,dtype=np.int)
        for i, t in enumerate(types):
            T[T_list==t] = i
    except:
        T = np.zeros(N)
    config.close()
    # assign values to Snapshot
    snap.N = N
    snap.xyz = R
    snap.L = L
    snap.T = T

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
    T = gsd_frame.particles.typeid
    # assign values to Snapshot
    snap.N = len(R)
    snap.xyz = R
    snap.L = L
    snap.T = T

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

def writeXYZ(filename,snap):
    fid = open(filename,'w+')
    print('%d'%snap.N,file=fid)
    print('Lattice="%.6f 0.0 0.0 0.0 %.6f 0.0 0.0 0.0 %.6f"'%tuple(snap.L),file=fid)
    for i in range(snap.N):
        print('C %.6f %.6f %.6f'%tuple(snap.xyz[i]),file=fid)
    fid.close()

def writeXML(filename,snap,bonds=False):
    fid = open(filename,'w+')
    print('<?xml version="1.0" encoding="UTF-8"?>',file=fid)
    print('<hoomd_xml version="1.7">',file=fid)
    print('<configuration time_step="0" dimensions="3" natoms="%d" >'%snap.N,file=fid)
    print('<box lx="%.6f" ly="%.6f" lz="%.6f" xy="0" xz="0" yz="0"/>'%tuple(snap.L),file=fid)
    print('<position num="%d">'%snap.N,file=fid)
    for i in range(snap.N):
        print('%.6f %.6f %.6f'%tuple(snap.xyz[i]),file=fid)
    print('</position>',file=fid)
    if bonds:
        bonds = []
        for i in range(snap.N):
            for j in snap.neighbors[i]:
                if i == j:
                    continue
                vec = snap.xyz[i] - snap.xyz[j]
                if np.all(vec == snap.wrap(vec)):
                    bonds.append( (i,j) )
        nb = len(bonds)
        print('<bond num="%d">'%nb,file=fid)
        for bond in bonds:
            print('backbone %d %d'%bond,file=fid)
        print('</bond>',file=fid)
    print('</configuration>',file=fid)
    print('</hoomd_xml>',file=fid)
    fid.close()
