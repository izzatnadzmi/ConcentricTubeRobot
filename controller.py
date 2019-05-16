'''
    Author: Izzat Kamarudzaman
    Source Version: 0.2.2
    Python Version: 3.7.2
    Adapted from Matlab code by Mohsen Khadem

    Code visualises three-tubed concentric tube continuum robot.
'''

# Import required packages 
# from __future__ import division, print_function
import os
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import csv
from scipy import stats
from collections import defaultdict
# from scipy.optimize import fmin_bfgs
from scipy.integrate import solve_ivp
from mpl_toolkits import mplot3d
# #matplotlib inline


## main ode solver
def moving_CTR(q,uz_0):

    l = 1e-3 * np.array([431, 332, 174])  # length of tubes
    l_k = 1e-3 * np.array([103, 113, 134])  # length of the curved part of tubes

    # physical parameters
    E = np.array([ 6.4359738368e+10, 5.2548578304e+10, 4.7163091968e+10])
    J = 1.0e-11 * np.array([0.0120, 0.0653, 0.1686])
    I = 1.0e-12 * np.array([0.0601, 0.3267, 0.8432])
    G = np.array([2.5091302912e+10, 2.1467424256e+10, 2.9788923392e+10] )
    Ux = np.array([21.3, 13.108, 3.5])
    Uy = np.array([0, 0, 0])
    n = 3

    # q1 o q3 are robot base movments, q3 to q6 are rbot base rotation angle.
    # if ~(iscolumn(uz_0))
    #     uz0=uz_0'
    # else
    #     uz0=uz_0
    # end
    uz0 = uz_0.copy()

    q_0 = np.array([-0.2858, -0.2025, -0.0945, 0, 0, 0]) # ' transposed previously
    B = np.array([q[0]+q_0[0], q[1]+q_0[1], q[2]+q_0[2]])  # +1  # length of tubes before template

    #initial angles
    alpha = np.array([q[3]+q_0[3], q[4]+q_0[4], q[5]+q_0[5]]) - B * uz0.transpose()  # +1  # ' transposed previously
    alpha = alpha.flatten()  # vectorise. check again
    alpha_1 = alpha[0]  # +1  # need? .copy()

    # segmenting tubes  
    # check all inputs must have n elements, n is number of tubes
    (L, d_tip, EE, UUx, UUy) = segmenting(E,Ux,Uy,l,B,l_k)

    # print('L: \n', L)
    # print('\nd_tip: \n', d_tip)
    # print('\nEE: \n', EE)
    # print('\nUUx: \n', UUx)
    # print('\nUUy: \n', UUy)

    SS = L.copy()

    for i in np.arange(len(L)):  # i=1:length(L):
        SS[i] = np.sum(L[:i+1])  # +1
    #     plot((B(1)+SS(i))*ones(1,10),1:10,'b' ,'LineWidth',2)

    # S is segmented abssica of tube after template
    S = SS[SS+np.min(B) > 0] + np.min(B)
    E = np.zeros((n, len(S)))
    Ux = E.copy()
    Uy = E.copy()
    for i in np.arange(n):  # i=1:n
        E[i,:] = EE[i,SS+np.min(B)>0]
        Ux[i,:] = UUx[i,SS+np.min(B)>0]
        Uy[i,:] = UUy[i,SS+np.min(B)>0]
     # each (i,j) element of above matrices correspond to the jth segment of
     # ith tube, 1st tube is the most inner
    # E
    # Ux
    ## Solving ode for shape

    span = np.hstack((0, S))       # vector of tube abssica starting at zero
    Length = np.array([], dtype=np.int64).reshape(0,1)
    r = np.array([], dtype=np.int64).reshape(0,3)
    U_z = np.array([], dtype=np.int64).reshape(0,3) # solved length, curvatures, and twist angles
    #U1_after=[0;0;0];             # 1st tube initial curvature at segment beginning
    r0 = np.array([[0, 0, 0]]).transpose()  # ' transpose?
    R0 = np.array([[np.cos(alpha_1), np.sin(alpha_1), 0],[-np.sin(alpha_1), np.cos(alpha_1), 0], [0, 0, 1]])
    R0 = R0.reshape(9,1,order='F')  # fortran scan order
    #alpha=alpha-B.*uz_0' 

    for seg in np.arange(len(S)):  # =1:length(S)

        s_span = [span[seg], span[seg+1]-0.0000001]
        y0_1 = np.vstack([r0, R0])

        y0_2 = np.zeros((2*n,1))
        y0_2[n:2*n] = np.reshape(alpha.copy(), (3,1))  # +1
        y0_2[0:n] = np.reshape(uz0.copy(), (3,1))

        y_0 = np.vstack([y0_2, y0_1]).flatten()

        EI = E[:,seg] * I.transpose()
        GJ = G * J
        ode_sols = solve_ivp(lambda s,y: ode(s,y,Ux[:,seg],Uy[:,seg],EI,GJ,n), s_span, y_0, method='RK23')
        (s,y) = (ode_sols.t[:, np.newaxis], ode_sols.y.transpose())
        # first n elements of y are curvatures along z, e.g., y= [ u1_z  u2_z ... ]
        # last n elements of y are twist angles, alpha_i
        shape = np.array([y[:,2*n], y[:,2*n+1], y[:,2*n+2]]).transpose()  # +1
        Length = np.vstack([Length, s])
        r = np.vstack([r, shape])
        U_z = np.vstack([U_z, y[:,0:n]])  # +1
        r0 = shape[-1][:, np.newaxis]  # (end,:)' reshape?
        R0 = y[-1, 2*n+3:2*n+12][:, np.newaxis]  # +1 (end,2*n+4:2*n+12)'
        uz0 = U_z.copy()[-1]  # (end,:)'  #transpose doesnt work

    Uz = np.zeros((n,1))
    for i in np.arange(n):
        index =  np.argmin(np.abs(Length-d_tip[i]+0.0001) )
        Uz[i] = U_z[index, i]  # copy?

    r1 = r.copy()
    tube2_end = np.argmin(np.abs(Length-d_tip[1]))
    r2 = np.array([r[0:tube2_end,0], r[0:tube2_end,1], r[0:tube2_end,2]]).transpose()
    tube3_end = np.argmin(np.abs(Length-d_tip[2]))
    r3 = np.array([r[0:tube3_end,0], r[0:tube3_end,1], r[0:tube3_end,2]]).transpose()  # +1

    return (r1, r2, r3, Uz)


## ODE # call -> (s,y) = ode23(@(s,y) ode(s,y,Ux(:,seg),Uy(:,seg),E(:,seg).*I',G.*J,n), s_span, y_0)
def ode(s, y, Ux, Uy, EI, GJ, n):  # dydt s>~

    dydt = np.zeros(2*n+12)
    # first n elements of y are curvatures along z, e.g., y= [ u1_z  u2_z ... ]
    # second n elements of y are twist angles, alpha_i
    # last 12 elements are r (position) and R (orientations), respectively
    # calculating 1st tube's curvatures in x and y direction
    ux = np.zeros((n,1))
    uy = np.zeros((n,1))

    # calculating tube's curvatures in x and y direction
    for i in np.arange(n):  # alpha to curvature
        ux[i] = (1/(EI[0]+EI[1]+EI[2])) * (
                EI[0]*Ux[0]*np.cos(y[n+i]-y[n+0]) + EI[0]*Uy[0]*np.sin(y[n+i]-y[n+0]) +
                EI[1]*Ux[1]*np.cos(y[n+i]-y[n+1]) + EI[1]*Uy[1]*np.sin(y[n+i]-y[n+1]) +
                EI[2]*Ux[2]*np.cos(y[n+i]-y[n+2]) + EI[2]*Uy[2]*np.sin(y[n+i]-y[n+2]) 
        )

        uy[i]= (1/(EI[0]+EI[1]+EI[2])) * (
                -EI[0]*Ux[0]*np.sin(y[n+i]-y[n+0]) + EI[0]*Uy[0]*np.cos(y[n+i]-y[n+0]) +
                -EI[1]*Ux[1]*np.sin(y[n+i]-y[n+1]) + EI[1]*Uy[1]*np.cos(y[n+i]-y[n+1]) +
                -EI[2]*Ux[2]*np.sin(y[n+i]-y[n+2]) + EI[2]*Uy[2]*np.cos(y[n+i]-y[n+2]) 
        )

    # odes for twist
    for i in np.arange(n):
        dydt[i] =  ((EI[i])/(GJ[i])) * (ux[i]*Uy[i] -  uy[i]*Ux[i] )  # ui_z
        dydt[n+i] =  y[i]   #alpha_i

    e3 = np.array([0, 0, 1]).transpose()  # '
    uz = y[0:n]

    # y(1) to y(3) are position of point materials
    #r1=[y(1) y(2) y(3)]
    # y(4) to y(12) are rotation matrix elements
    R1 = np.array([ [y[2*n+3], y[2*n+4], y[2*n+5]], 
                    [y[2*n+6], y[2*n+7], y[2*n+8]], 
                    [y[2*n+9], y[2*n+10], y[2*n+11]] ])  # +1

    u_hat = np.array([  [0, -uz[0], uy[0]],
                        [uz[0], 0, -ux[0]],
                        [-uy[0], ux[0], 0]])

    # odes
    dr1 = R1@e3
    dR1 = R1@u_hat

    dydt[2*n+0] = dr1[0]
    dydt[2*n+1] = dr1[1]
    dydt[2*n+2] = dr1[2]  # +1
    dR = dR1.flatten() # ' 
    for i in np.arange(3, 12):  # +1  # =4:12
        dydt[2*n+i] = dR[i-3]

    return dydt


## code for segmenting tubes
def segmenting(E,Ux,Uy,l,B,l_k):  # -> [L,d1,E,Ux,Uy,I,G,J]

    # all vectors must be sorted, starting element belongs to the most inner tube
    #E, U, I, G, J   stifness, curvature, inertia, torsion constant, and second moment of inertia vectors for each tube
    #l vector of tube length
    #B  vector of tube movments with respect to template position, i.e., s=0 (always negative)
    #l_k vecot oftube's curved part length

    d1 = l+B # position of tip of the tubes
    d2 = d1-l_k # position of the point where tube bending starts
    points = np.hstack((0, B, d2, d1))
    index = np.argsort(points)  # [L, index]=sort(points)
    L = points[index]
    L = 1e-5*np.floor(1e5*np.diff(L))  # length of each segment 
    #(used floor because diff command doesn't give absolute zero sometimes)
    # 
    # for i=1:k-1
    # if B(i)>B(i+1)
    #     sprintf('inner tube is clashing into outer tubes')
    #     E=zeros(k,length(L))
    #     I=E G=E J=E Ux=E Uy=E
    #     return
    # end
    # end

    EE = np.zeros((3,len(L)))
    II = EE.copy()
    GG = EE.copy()
    JJ = EE.copy()
    UUx = EE.copy()
    UUy = EE.copy()

    for i in np.arange(3): # 1:3

        a = np.argmin(np.abs(index-i+1)) # find where tube begins
        b = np.argmin(np.abs(index-(1*3+i+1))) # find where tube curve starts
        c = np.argmin(np.abs(index-(2*3+i+1))) # find where tube ends

        if L[a]==0:
            a=a+1
        if L[b]==0:
            b=b+1
        if c<len(L):  # <= matlab
            if L[c]==0:
                c=c+1
            
        EE[i,a:c]=E[i]
        UUx[i,b:c]=Ux[i]
        UUy[i,b:c]=Uy[i]

    l = L[np.nonzero(L)]  # ~(L==0)]  # get rid of zero lengthes
    E = np.zeros((3,len(l)))
    Ux = E.copy()
    Uy = E.copy()   # length https://stackoverflow.com/questions/30599101/translating-mathematical-functions-from-matlab-to-python
    for i in np.arange(3):  # =1:3
        E[i,:] = EE[i,~(L==0)]
        Ux[i,:] = UUx[i,~(L==0)] 
        Uy[i,:] = UUy[i,~(L==0)]

    L = L[np.nonzero(L)]  # (~(L==0))

    # L,d1,E,Ux,Uy,I,G,J
    return (L, d1, E, Ux, Uy)


if __name__ == "__main__":

    # tic
    # initial value of twist
    uz_0 = np.array([[0, 0, 0]]).transpose()
    q = np.array([0, 0, 0, 0, 0, 0])  #inputs
    (r1,r2,r3,Uz) = moving_CTR(q, uz_0)
    # toc

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(r1[:,0], r1[:,1], r1[:,2], linewidth=1, label="tube1")
    ax.plot3D(r2[:,0], r2[:,1], r2[:,2], linewidth=2)
    ax.plot3D(r3[:,0], r3[:,1], r3[:,2], linewidth=3)
    q = np.array([0, 0, 0, np.pi, 0, 0])  #inputs
    (r1,r2,r3,Uz) = moving_CTR(q, uz_0)
    ax.plot3D(r1[:,0], r1[:,1], r1[:,2], linewidth=1, label="tube2")
    ax.plot3D(r2[:,0], r2[:,1], r2[:,2], linewidth=2)
    ax.plot3D(r3[:,0], r3[:,1], r3[:,2], linewidth=3)
    q = np.array([0, 0, 0, 0, np.pi, 0])  #inputs
    (r1,r2,r3,Uz) = moving_CTR(q, uz_0)
    ax.plot3D(r1[:,0], r1[:,1], r1[:,2], linewidth=1, label="tube3")
    ax.plot3D(r2[:,0], r2[:,1], r2[:,2], linewidth=2)
    ax.plot3D(r3[:,0], r3[:,1], r3[:,2], linewidth=3)
    q = np.array([0, 0, 0, 0, 0, np.pi])  #inputs
    (r1,r2,r3,Uz) = moving_CTR(q, uz_0)
    ax.plot3D(r1[:,0], r1[:,1], r1[:,2], linewidth=1, label="tube4")
    ax.plot3D(r2[:,0], r2[:,1], r2[:,2], linewidth=2)
    ax.plot3D(r3[:,0], r3[:,1], r3[:,2], linewidth=3)

    ax.legend()
    plt.show()
