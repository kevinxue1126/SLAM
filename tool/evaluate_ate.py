#!/usr/bin/python
# -*- coding:utf-8 -*-
# The absolute difference between the estimated estimated pose trajectory and the ground truth pose trajectory
# Software License Agreement (BSD License)
#
# Copyright (c) 2013, Juergen Sturm, TUM
# All rights reserved.

# Rely Requirements: 
# sudo apt-get install python-argparse

# Usage 
# python evaluate_ate.py gt.txt est.txt
# --plot3D Draw a 3D trajectory matching map
# --verbose Displays rms, mean, median, standard deviation, maximum and minimum values for all error information

import sys
import numpy
import argparse
import associate

# Similarity Transformation Error
def align_sim3(model, data):
    """Implementation of the paper: S. Umeyama, Least-Squares Estimation
       of Transformation Parameters Between Two Point Patterns,
       IEEE Trans. Pattern Anal. Mach. Intell., vol. 13, no. 4, 1991.
       Input:
           model -- first trajectory (3xn)
           data -- second trajectory (3xn)
       Output:
           s -- scale factor (scalar)
           R -- rotation matrix (3x3)
           t -- translation vector (3x1)
           t_error -- translational error per point (1xn)
    """
    # substract mean
    mu_M = model.mean(0).reshape(model.shape[0],1)
    mu_D = data.mean(0).reshape(data.shape[0],1)
    model_zerocentered = model - mu_M
    data_zerocentered = data - mu_D
    n = np.shape(model)[0]
    
    # correlation
    C = 1.0/n*np.dot(model_zerocentered.transpose(), data_zerocentered)
    sigma2 = 1.0/n*np.multiply(data_zerocentered,data_zerocentered).sum()
    U_svd,D_svd,V_svd = np.linalg.linalg.svd(C)
    D_svd = np.diag(D_svd)
    V_svd = np.transpose(V_svd)
    S = np.eye(3)

    if(np.linalg.det(U_svd)*np.linalg.det(V_svd) < 0):
    S[2,2] = -1

    R = np.dot(U_svd, np.dot(S, np.transpose(V_svd)))
    s = 1.0/sigma2*np.trace(np.dot(D_svd, S))
    t = mu_M-s*np.dot(R,mu_D)
    # TODO:
    model_aligned = s * R * model + t
    alignment_error = model_aligned - data
    t_error = np.sqrt(np.sum(np.multiply(alignment_error,alignment_error),0)).A[0]

    #return s, R, t #, t_error
    return s, R, t, t_erro

# Euclidean Transform Error
"""
Since the coordinate system of the real trajectory recording is different from the coordinate system at the beginning of the algorithm, 
there is a Euclidean transformation between the camera trajectory estimated by the algorithm and the real trajectory. 
An Euclidean transformation between the value and the matched estimate.

The estimated value is transformed and then the difference is calculated with the true value.

"""
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
        # outer() The former parameter indicates that the latter parameter expands the multiple
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

def plot_traj(ax,stamps,traj,style,color,label):
    """
    Plot a trajectory using matplotlib. 
    2D map
    Input:
    ax -- the plot
    stamps -- time stamps (1xn)
    traj -- trajectory (3xn)
    style -- line style
    color -- line color
    label -- plot legend
    
    """
    stamps.sort()
    interval = numpy.median([s-t for s,t in zip(stamps[1:],stamps[:-1])])
    x = []
    y = []
    last = stamps[0]
    for i in range(len(stamps)):
        if stamps[i]-last < 2*interval:
            x.append(traj[i][0])
            y.append(traj[i][1])
        elif len(x)>0:
            ax.plot(x,y,style,color=color,label=label)
            label=""
            x=[]
            y=[]
        last= stamps[i]
    if len(x)>0:
        ax.plot(x,y,style,color=color,label=label)
 
def plot_traj3D(ax,stamps,traj,style,color,label):
    """
    Plot a trajectory using matplotlib. 
    3D map
    Input:
    ax -- the plot
    stamps -- time stamps (1xn)
    traj -- trajectory (3xn)
    style -- line style
    color -- line color
    label -- plot legend
    
    """
    stamps.sort()
    interval = numpy.median([s-t for s,t in zip(stamps[1:],stamps[:-1])])
    x = []
    y = []
    z = []
    last = stamps[0]
    for i in range(len(stamps)):
        if stamps[i]-last < 2*interval:
            x.append(traj[i][0])
            y.append(traj[i][1])
            z.append(traj[i][2])
        elif len(x)>0:
            ax.plot(x,y,z,style,color=color,label=label)
            
            label=""
            x=[]
            y=[]
            z=[]
        last= stamps[i]
    if len(x)>0:
        ax.plot(x,y,z,style,color=color,label=label)          

if __name__=="__main__":
    # parse command line
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='''
    This script computes the absolute trajectory error from the ground truth trajectory and the estimated trajectory. 
    ''')
    parser.add_argument('first_file', help='ground truth trajectory (format: timestamp tx ty tz qx qy qz qw)')
    parser.add_argument('second_file', help='estimated trajectory (format: timestamp tx ty tz qx qy qz qw)')
    parser.add_argument('--offset', help='time offset added to the timestamps of the second file (default: 0.0)',default=0.0)
    parser.add_argument('--scale', help='scaling factor for the second trajectory (default: 1.0)',default=1.0)
    parser.add_argument('--max_difference', help='maximally allowed time difference for matching entries (default: 0.02)',default=0.02)
    parser.add_argument('--save', help='save aligned second trajectory to disk (format: stamp2 x2 y2 z2)')
    parser.add_argument('--save_associations', help='save associated first and aligned second trajectory to disk (format: stamp1 x1 y1 z1 stamp2 x2 y2 z2)')
    parser.add_argument('--plot', help='plot the first and the aligned second trajectory to an image (format: png)')
    parser.add_argument('--plot3D', help='plot the first and the aligned second trajectory to as interactive 3D plot (format: png)', action = 'store_true')
    parser.add_argument('--verbose', help='print all evaluation data (otherwise, only the RMSE absolute translational error in meters after alignment will be printed)', action='store_true')
    args = parser.parse_args()
    
    # read data
    first_list = associate.read_file_list(args.first_file)
    second_list = associate.read_file_list(args.second_file)
    
    # Match according to timestamp, the maximum difference cannot exceed max_difference 0.02
    # 
    matches = associate.associate(first_list, second_list,float(args.offset),float(args.max_difference))    
    if len(matches)<2:
        sys.exit("Couldn't find matching timestamp pairs between groundtruth and estimated trajectory! Did you choose the correct sequence?")
    
    # By matching pair
    first_xyz = numpy.matrix([[float(value) for value in first_list[a][0:3]] for a,b in matches]).transpose()
    second_xyz = numpy.matrix([[float(value)*float(args.scale) for value in second_list[b][0:3]] for a,b in matches]).transpose()
    
    # Solve the error for the matched position coordinates
    rot,trans,trans_error = align(second_xyz,first_xyz)
    
    # Mutual match error bars  
    second_xyz_aligned = rot * second_xyz + trans
    first_stamps = first_list.keys()
    first_stamps.sort()
    first_xyz_full = numpy.matrix([[float(value) for value in first_list[b][0:3]] for b in first_stamps]).transpose()
    second_stamps = second_list.keys()
    second_stamps.sort()
    second_xyz_full = numpy.matrix([[float(value)*float(args.scale) for value in second_list[b][0:3]] for b in second_stamps]).transpose()
    second_xyz_full_aligned = rot * second_xyz_full + trans
    
    if args.verbose:
        print "compared_pose_pairs %d pairs"%(len(trans_error))
        # Root mean square error
        print "absolute_translational_error.rmse %f m"%numpy.sqrt(numpy.dot(trans_error,trans_error) / len(trans_error))
        # Error mean
        print "absolute_translational_error.mean %f m"%numpy.mean(trans_error)
        # Median error
        print "absolute_translational_error.median %f m"%numpy.median(trans_error)
        # Standard deviation of error
        print "absolute_translational_error.std %f m"%numpy.std(trans_error)
        # Error minimum
        print "absolute_translational_error.min %f m"%numpy.min(trans_error)
        # Error maximum
        print "absolute_translational_error.max %f m"%numpy.max(trans_error)
    else:
        print "%f"%numpy.sqrt(numpy.dot(trans_error,trans_error) / len(trans_error))
        
    if args.save_associations:
        file = open(args.save_associations,"w")
        file.write("\n".join(["%f %f %f %f %f %f %f %f"%(a,x1,y1,z1,b,x2,y2,z2) for (a,b),(x1,y1,z1),(x2,y2,z2) in zip(matches,first_xyz.transpose().A,second_xyz_aligned.transpose().A)]))
        file.close()
        
    if args.save:
        file = open(args.save,"w")
        file.write("\n".join(["%f "%stamp+" ".join(["%f"%d for d in line]) for stamp,line in zip(second_stamps,second_xyz_full_aligned.transpose().A)]))
        file.close()

    if args.plot:
        # Draw 2D diagrams
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.pylab as pylab
        from matplotlib.patches import Ellipse
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plot_traj(ax,first_stamps,first_xyz_full.transpose().A,'-',"black","ground truth")
        plot_traj(ax,second_stamps,second_xyz_full_aligned.transpose().A,'-',"blue","estimated")
        
        # Error bars
        #label="difference"
        #for (a,b),(x1,y1,z1),(x2,y2,z2) in zip(matches,first_xyz.transpose().A,second_xyz_aligned.transpose().A):
        #    ax.plot([x1,x2],[y1,y2],'-',color="red",label=label)
        #    label=""
            
        ax.legend()
            
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        plt.savefig(args.plot,dpi=90)
        
    if args.plot3D:
        # Draw 3D diagrams
        import matplotlib as mpl
        mpl.use('Qt4Agg')
        from mpl_toolkits.mplot3d import Axes3D
        import numpy as np
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        #        ax = fig.add_subplot(111)
        plot_traj3D(ax,first_stamps,first_xyz_full.transpose().A,'-',"black","groundTruth")
        plot_traj3D(ax,second_stamps,second_xyz_full_aligned.transpose().A,'-',"blue","orb-slam2-flow")
        
        # Error bars
        #label="difference"
        #for (a,b),(x1,y1,z1),(x2,y2,z2) in zip(matches,first_xyz.transpose().A,second_xyz_aligned.transpose().A):
        #    ax.plot([x1,x2],[y1,y2],[z1,z2],'-',color="red",label=label)
        #    label=""            
        
        ax.legend()
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        print "Showing"
        plt.show(block=True)
        plt.savefig("./test.png",dpi=90)
        #answer = raw_input('Back to main and window visible? ')
        #if answer == 'y':
        #    print('Excellent')
        #else:
        #    print('Nope')

    #plt.savefig(args.plot,dpi=90)
