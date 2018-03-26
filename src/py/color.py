#
# color.py
# assign colors to particles based on position in transformed manifold
#
# Copyright (c) 2018 Wesley Reinhart.
# This file is part of the crayon project, released under the Modified BSD License.

from __future__ import print_function

import numpy as np

def compressColors(colors,delta=0.001):
    # obtain colors which will fit in the VMD color space
    c = np.vstack({tuple(row) for row in np.floor(colors/delta)*delta})
    while len(c) > 1024:
        delta = delta * 2
        c = np.vstack({tuple(row) for row in np.floor(colors/delta)*delta})
    # make NaN colors black
    nan_idx = np.argwhere(c[:,0] != c[:,0]).flatten()
    c[nan_idx,:] = 0.
    # map the individual colors into the compressed color space
    cmap = np.zeros(len(colors))
    for i, row in enumerate(colors):
        if np.any(row != row):
            cmap[i] = -1
            continue
        colors_round = np.ones((c.shape[0],1)) * np.floor(row/delta)*delta
        color_delta = np.sum( (colors_round - c )**2., 1)
        cmap[i] = np.argwhere(color_delta == 0).flatten()[0]
    return c, cmap

def neighborSimilarity(f_map,neighbors,coords):
    N = len(f_map)
    f_dat = np.zeros((N,3))
    for i in range(N):
        nn = np.array(f_map[np.asarray(neighbors[i])],dtype=np.int)
        delta = np.sqrt(np.sum((coords[nn,:] - coords[f_map[i],:])**2.,1))
        f_dat[i,0] = np.mean(delta)
        f_dat[i,1] = np.min(delta)
    for i in range(N):
        nn = np.array(f_map[np.asarray(neighbors[i])],dtype=np.int)
        f_dat[i,2] = np.mean(f_dat[nn,0])
    return f_dat

def writeVMD(filename,snapshots,colors,com,n_col,sigma=1.0,swap=('',''),mode='add',suffix=''):
    # create a VMD draw script
    fid = open(filename,'w')
    cmds = ['axes location off',
            'display rendermode GLSL',
            'display projection orthographic',
            'display ambientocclusion on',
            'display aodirect 0.60',
            'display cuedensity 0.001',
            'color Display Background white']
    for cmd in cmds:
        print(cmd,file=fid)
    s = 33
    e = 1057
    ncol = e - s
    c = s
    for i in range(len(colors)):
        rgb = colors[i,:]
        print("color change rgb %d %.3f %.3f %.3f"%(c,rgb[0],rgb[1],rgb[2]),file=fid)
        c += 1
    prev = 'None'
    for f, frame in enumerate(snapshots):
        xml_prefix = frame.replace(swap[0],swap[1])
        frame_run = frame.split('.')[0]
        if frame_run != prev or mode == "new":
            if '.xml' in xml_prefix:
                on_load = 'type hoomd'
            else:
                on_load = ''
            print('mol new %s %s'%(xml_prefix,on_load),file=fid)
            newFrame = True
        else:
            print('mol addfile %s %s'%(xml_prefix,on_load),file=fid)
            newFrame = False
        cmds = ['[atomselect top "all"] set radius %f'%(0.50*sigma),
                'set fid [open "%s'%xml_prefix + '_%d%d%d'%com + '%s.cmap"]'%suffix,
                'set file_data [read $fid]',
                'close $fid',
                'set sel [atomselect top "all"]',
                'set userVals {"user" "user2" "user3" "user4"}',
                'for {set j 0} {$j < %d} {incr j} {'%n_col,
                '    set v {}',
                '    for {set i $j} {$i < [llength $file_data]} {incr i %d} {'%n_col,
                '        lappend v [lindex $file_data $i]',
                '    }',
                '    $sel set [lindex $userVals $j] $v',
                '}']
        for cmd in cmds:
            print(cmd,file=fid)
        if newFrame:
            cmds = ['mol rename top %s'%frame_run.split('/')[-1],
                    'mol modstyle 0 top vdw 1.0 25',
                    'mol modmaterial 0 top "AOChalky"',
                    'mol modselect 0 top "all"',
                    'mol modcolor 0 top ColorID 6',
                    'mol selupdate 0 top 1',
                    'mol colupdate 0 top 1',
                    'mol addrep top',
                    'mol modstyle 1 top vdw 1.0 25',
                    'mol modmaterial 1 top "AOChalky"',
                    'mol modselect 1 top "user >= 0"',
                    'mol modcolor 1 top user',
                    'mol scaleminmax top 1 0 1023',
                    'mol selupdate 1 top 1',
                    'mol colupdate 1 top 1',
                    'mol off top']
            for cmd in cmds:
                print(cmd,file=fid)
        prev = frame_run
    fid.close()
