#!/usr/bin/env python
# coding=utf-8
# Draw camera 3D trajectory
# Use python draw_groundtruth_Rt_associate.py
import numpy
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d

import sys

# data match
import associate



def align(model,data):
    """Align two trajectories using the method of Horn (closed-form).
    Matching Error Calculation
    Input:
    model -- first trajectory (3xn)    estimated value
    data -- second trajectory (3xn)    truth value
    
    Output:
    rot -- rotation matrix (3x3)    Rotation and translation matrix of two data
    trans -- translation vector (3x1)
    trans_error -- translational error per point (1xn) match error
    
    """
    numpy.set_printoptions(precision=3,suppress=True)
    model_zerocentered = model - model.mean(1) # de-mean
    data_zerocentered = data - data.mean(1)
    
    W = numpy.zeros( (3,3) )# 
    for column in range(model.shape[1]):
        W += numpy.outer(model_zerocentered[:,column],data_zerocentered[:,column])
        # outer() The former parameter indicates the expansion multiple of the latter parameter
        # https://blog.csdn.net/hqh131360239/article/details/79064592
    U,d,Vh = numpy.linalg.linalg.svd(W.transpose())# singular value decomposition
    S = numpy.matrix(numpy.identity( 3 ))# unit array
    if(numpy.linalg.det(U) * numpy.linalg.det(Vh)<0):
        S[2,2] = -1
    rot = U*S*Vh
    trans = data.mean(1) - rot * model.mean(1)
    
    model_aligned = rot * model + trans
    alignment_error = model_aligned - data
    # err = sqrt((x-x')^2 + (y-y')^2 + (z-z')^2) 
    trans_error = numpy.sqrt(numpy.sum(numpy.multiply(alignment_error,alignment_error),0)).A[0]
        
    return rot,trans,trans_error


# read data 
ground_truth_list = associate.read_file_list("./groundtruth.txt")   # real trajectory data
src_list          = associate.read_file_list("./src.txt")           # Raw orb-slam2 prediction
flow_list         = associate.read_file_list("./flow3.txt")         # flow optical flow blessing
geom_list         = associate.read_file_list("./geom.txt")          # geom multi-view geometry blessing
offset = 0.0 # time offset
max_difference = 0.02 # maximum time difference
scale = 1.0 # data size scale 


# The actual value is compared with the predicted value of the original version of the algorithm
matches_src = associate.associate(ground_truth_list, src_list,float(offset),float(max_difference))    
if len(matches_src)<2:
    sys.exit("Couldn't find matching timestamp pairs between groundtruth and estimated trajectory! Did you choose the correct sequence?")

# By matching pair
gt_src_ass_xyz = numpy.matrix([[float(value) for value in ground_truth_list[a][0:3]] for a,b in matches_src]).transpose()
src_ass_xyz = numpy.matrix([[float(value) * float(scale) for value in src_list[b][0:3]] for a,b in matches_src]).transpose()
# Solve the error for the matched position coordinates
rot,trans,trans_error = align(src_ass_xyz,gt_src_ass_xyz)
# All values ​​of the original data
gt_stamps = ground_truth_list.keys()
gt_stamps.sort() # Sort by chronological order
gt_xyz = numpy.matrix([[float(value) for value in ground_truth_list[b][0:3]] for b in gt_stamps]).transpose()
src_stamps = src_list.keys()
src_stamps.sort()
src_xyz = numpy.matrix([[float(value) * float(scale) for value in src_list[b][0:3]] for b in src_stamps]).transpose()
src_xyz_aligned = rot * src_xyz + trans # value after euclidean transformation
print "src: %d pairs"%(len(trans_error))
# root mean square error
print "src rmse : %f m"%numpy.sqrt(numpy.dot(trans_error,trans_error) / len(trans_error))




# The actual value is compared with the predicted value of the flow version algorithm
matches_flow = associate.associate(ground_truth_list, flow_list,float(offset),float(max_difference))    
if len(matches_flow)<2:
    sys.exit("Couldn't find matching timestamp pairs between groundtruth and estimated trajectory! Did you choose the correct sequence?")

# By matching pair
gt_flow_ass_xyz = numpy.matrix([[float(value) for value in ground_truth_list[a][0:3]] for a,b in matches_flow]).transpose()
flow_ass_xyz = numpy.matrix([[float(value) * float(scale) for value in flow_list[b][0:3]] for a,b in matches_flow]).transpose()
# Solve the error for the matched position coordinates
rot2,trans2,trans_error2 = align(flow_ass_xyz,gt_flow_ass_xyz)

# All values ​​of the original data
flow_stamps = flow_list.keys()
flow_stamps.sort() # Sort in chronological order
flow_xyz = numpy.matrix([[float(value) * float(scale) for value in flow_list[b][0:3]] for b in flow_stamps]).transpose()
flow_xyz_aligned = rot2 * flow_xyz + trans2 # value after euclidean transformation
print "flow: %d pairs"%(len(trans_error2))
# root mean square error
print "flow rmse : %f m"%numpy.sqrt(numpy.dot(trans_error2,trans_error2) / len(trans_error2))





# The actual value is compared with the predicted value of the geom version of the algorithm
matches_geom = associate.associate(ground_truth_list, geom_list,float(offset),float(max_difference))    
if len(matches_geom)<2:
    sys.exit("Couldn't find matching timestamp pairs between groundtruth and estimated trajectory! Did you choose the correct sequence?")

# By matching pair
gt_geom_ass_xyz = numpy.matrix([[float(value) for value in ground_truth_list[a][0:3]] for a,b in matches_geom]).transpose()
geom_ass_xyz = numpy.matrix([[float(value) * float(scale) for value in geom_list[b][0:3]] for a,b in matches_geom]).transpose()
# Solve the error for the matched position coordinates
rot3,trans3,trans_error3 = align(geom_ass_xyz,gt_geom_ass_xyz)

# All values ​​of the original data
geom_stamps = geom_list.keys()
geom_stamps.sort() # Sort in chronological order
geom_xyz = numpy.matrix([[float(value) * float(scale) for value in geom_list[b][0:3]] for b in geom_stamps]).transpose()
geom_xyz_aligned = rot3 * geom_xyz + trans3 # value after euclidean transformation
print "geom: %d pairs"%(len(trans_error3))
# root mean square error
print "geom rmse : %f m"%numpy.sqrt(numpy.dot(trans_error3,trans_error3) / len(trans_error3))


# Convert 3 rows × n columns into an array array
gt_xyz = gt_xyz.A
src_xyz_aligned = src_xyz_aligned.A
flow_xyz_aligned = flow_xyz_aligned.A
geom_xyz_aligned = geom_xyz_aligned.A
# 
#print type(gt_xyz)
#print gt_xyz
#Sprint gt_xyz[0]

ax = plt.subplot( 111, projection='3d')
ax.plot(gt_xyz[0], gt_xyz[1], gt_xyz[2],'r',label='groundTruth')  # red lines
ax.plot( src_xyz_aligned[0], src_xyz_aligned[1],src_xyz_aligned[2], 'b',label='orb-slam2-src')  # blue line
ax.plot(flow_xyz_aligned[0],flow_xyz_aligned[1],flow_xyz_aligned[2],'g',label='orb-slam2-flow')  # green line
ax.plot(geom_xyz_aligned[0],geom_xyz_aligned[1],geom_xyz_aligned[2],'y',label='orb-slam2-geom')  # yellow line

#ax.plot(gt_xyz.transpose(),'r',label='groundTruth')  # red lines
#ax.plot(src_xyz_aligned.transpose(),'b',label='orb-slam2-src')  # blue line
#ax.plot(flow_xyz_aligned.transpose(),'g',label='orb-slam2-flow')  # green line
#ax.plot(geom_xyz_aligned.transpose(),'y',label='orb-slam2-geom')  # yellow line

ax.legend(loc='upper center')# show legend label=' '  ‘center right'  best upper center
ax.set_zlabel('Z/m')  # Axis
ax.set_ylabel('Y/m')
ax.set_xlabel('X/m')
ax.set_title('trajectory')#Figure title
plt.xlim( -1.6, 0.2 )
ax.set_zlim( 0.4, 2.6 )

plt.show()
