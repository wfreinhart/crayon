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
    # map the individual colors into the compressed color space
    cmap = np.zeros(len(colors))
    for i, row in enumerate(colors):
        colors_round = np.ones((c.shape[0],1)) * np.floor(row/delta)*delta
        color_delta = np.sum( (colors_round - c )**2., 1)
        cmap[i] = np.argwhere(color_delta == 0).flatten()[0]
    return c, cmap

# def colorMap(Ensemble):

# Y = np.loadtxt('vor_transform.dat')
# colors = np.loadtxt('vor_color_map.dat')

# # get particle graph identity
# u = 0
# frame_idx = np.zeros(N)
# for i in range(N):
#     if q6filter is True and Q6[i] < 0.20:
#         if flag_active:
#             print('Q6 FILTER APPLIED')
#             flag_active = False
#         continue
#     A = build_adjacency(i,NN)
#     sig = sig_from_A(A)
#     try:
#         idx = sig_map[ sig ]
#         frame_idx[i] = idx
#     except:
#         print('warning: could not find idx for particle %d'%i)
#         u += 1
# if u > 0:
#     print('warning: left %d particles unassigned!'%u)

# # check similarity to particles in neighbor shell
# s = np.zeros((N,3))
# for i in range(N):
#     nnbr  = np.array(frame_idx[np.asarray(NN[i])],dtype=np.int)
#     nevec = Y[nnbr,:]
#     n = len(nnbr)
#     devec = np.sqrt(np.sum((nevec - Y[frame_idx[i]])**2.,1))
#     s[i,0] = np.mean(devec)
#     s[i,1] = np.min(devec)
# for i in range(N):
#     nnbr  = np.asarray(NN[i],dtype=np.int)
#     s[i,2] = np.mean(s[nnbr,0])
# # assign color to each particle
# frame_colors = np.zeros((N,s.shape[1]+1)) - 1
# for i in range(N):
#     try:
#         frame_colors[i,0] = int( cmap[frame_idx[i]] )
#         frame_colors[i,1:] = s[i,:]
#     except:
#         print('warning: could not find frame color for particle %d (idx = %d, N = %d)'%(i,idx,len(cmap.keys())))

# # save dists so VMD can read them
# np.savetxt(frame.replace('.xml','.cmap'),frame_colors,fmt='%f')

def writeVMD(filename,snapshots,colors,n_col,sigma=1.0,file_type='hoomd',swap=('',''),mode='add'):
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
            print('mol new %s type %s'%(xml_prefix,file_type),file=fid)
            newFrame = True
        else:
            print('mol addfile %s type %s'%(xml_prefix,file_type),file=fid)
            newFrame = False
        cmds = ['[atomselect top "all"] set radius %f'%(0.50*sigma),
                'set fid [open "%s.cmap"]'%xml_prefix,
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
            cmds = ['mol rename top %s'%frame_run.split('/')[-2],
                    'mol modstyle 0 top vdw 1.0 25',
                    'mol modmaterial 0 top "AOChalky"',
                    'mol modselect 0 top "all"',
                    'mol modcolor 0 top ColorID 8',
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
