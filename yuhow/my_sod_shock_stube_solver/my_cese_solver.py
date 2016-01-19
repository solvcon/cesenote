"""
Sod Shock Tube Problem -- CESE solver v2
Author: You-Hao Chang

Journal of Computational Physics 119, 295-324 (1995)
The Method of Space-Time Conservation Element and Solution
Element -- A New Approach for Solving the Navier-Stokes and 
Eluer Equations

Sod tube condition: 
    x range: 102 points btw -0.505 ~ 0.505
    two region: left region, -0.505 ~ -0.005 (51 points)
                right region, 0.005 ~  0.505 (51 points)
"""

import matplotlib
matplotlib.use('TKAgg')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# global variable
it = 100            # number of iterations
npx = it + 2        # number of points (x-axis)
dt = 0.4 * 10**(-2) # time interval = 0.004
dx = 0.1 * 10**(-1) # space interval = 0.01
ga = 1.4            # gamma = Cp/Cv

rhol = 1.0
pl   = 1.0
vl   = 0.0
rhor = 0.125
pr   = 0.1
vr   = 0.0

weight = 1 # weighting factor of Eq.(4.39) 

# necessary matrices for CESE calculation
mtx_q  = np.asmatrix(np.zeros(shape=(3, npx)))
mtx_qn = np.asmatrix(np.zeros(shape=(3, npx)))
mtx_qx = np.asmatrix(np.zeros(shape=(3, npx)))
mtx_f  = np.asmatrix(np.zeros(shape=(3, 3)))
mtx_qt = np.asmatrix(np.zeros(shape=(3, npx)))
mtx_s  = np.asmatrix(np.zeros(shape=(3, npx)))
uxl = np.zeros(shape=(3,1))
uxr = np.zeros(shape=(3,1))

# output lists
xx  = [0. for idx in range(npx)] # x-axis
status_rho  = [rhor if idx >= npx / 2 else rhol for idx in range(npx)] # rho
status_vel  = [0. for idx in range(npx)] # v
status_p  = [pr if idx >= npx / 2 else pl for idx in range(npx)] # p

# initialization of matrix u
for j in xrange(0, npx):
    # Eq (4.1), u1, u2 and u3
    if j < npx / 2:
        mtx_q[0, j] = rhol
        mtx_q[1, j] = rhol * vl
        mtx_q[2, j] = pl / (ga - 1.0) + 0.5 * rhol * vl**2
    else:
        mtx_q[0, j] = rhor
        mtx_q[1, j] = rhor * vr
        mtx_q[2, j] = pr / (ga - 1.0) + 0.5 * rhor * vr**2

# setting the x-axis, from -0.505 to 0.505
xx[0] = -0.5 * dx * float(it + 1)
for j in xrange(0, npx - 1):
    xx[j+1] = xx[j] + dx

# initialization of output plots
fig = plt.figure()
frame_seq = []

# variables of looping 
mtx_length = npx
start_point = npx / 2 - 1
stepping = 2

# start to evaluate the solution iteratively
for i in xrange(0, it):

    # evaluate the current status of gas
    #for j in xrange(0, mtx_length):
    for j in xrange(start_point, start_point + stepping):

        # Eq. (4.7): constructing the matrix fm,k 
        # other reference: Yung-Yu's notes -> Sec.3.1, Eq. (3.14)
        w2 = mtx_q[1, j] / mtx_q[0, j]  # u2/u1
        w3 = mtx_q[2, j] / mtx_q[0, j]  # u3/u1
        mtx_f[0, 0] = 0.0
        mtx_f[0, 1] = 1.0
        mtx_f[0, 2] = 0.0
        mtx_f[1, 0] = -0.5 * (3.0 - ga) * w2**2
        mtx_f[1, 1] = (3.0 - ga) * w2
        mtx_f[1, 2] = ga - 1.0
        mtx_f[2, 0] = (ga - 1.0) * w2**3 - ga * w2 * w3
        mtx_f[2, 1] = ga * w3 - 1.5 * (ga - 1.0) * w2**2 
        mtx_f[2, 2] = ga * w2

        # Eq.(4.17), (u_mt)nj = -(f_mx)nj = -(fm,k*u_kx)nj, (u_mt)nj -> qt, (u_kx)nj -> qx
        mtx_qt[:,j] = -1.0 * mtx_f * mtx_qx[:,j]

        # Eq.(4.25), (u_m)nj -> q, (u_mt)nj -> qt, (u_kx)nj -> qx
        # for s0: u_0x -> mtx_qx[0, j], f_0 -> mtx_q[1, j], f_0t -> f0,k*u_kt 
        #     s1: u_1x -> mtx_qx[1, j], f_1 -> mtx_f[1, 2]*mtx_q[2, j]+mtx_f[1, 0]*mtx_q[0, j]+mtx_f[1, 1]*mtx_q[1, j]
        #                               f_1t-> mtx_f[1, 0]*mtx_qx[0, j]+mtx_f[1, 1]*mtx_qx[1, j]+mtx_f[1, 2]*mtx_qx[2, j]
        #     s2: u_2x -> mtx_qx[2, j], f_2 -> mtx_f[2, 0]*mtx_q[0, j]+mtx_f[2, 1]*mtx_q[1, j]+mtx_f[2, 2]*mtx_q[2, j]
        #                               f_2t-> *mtx_qx[0, j]+f32*mtx_qx[1, j]+f33*mtx_qx[2, j]
        mtx_s[:, j] = 0.25 * dx * mtx_qx[:, j] + (dt / dx) * mtx_f * mtx_q[:,j] - 0.25 * dt * (dt / dx) * mtx_f * mtx_f * mtx_qx[:,j]

    # evaluate the status of gas after time stamp moves forward by 1 dt/2
    #mm = mtx_length - 1
    #for j in xrange(0, mm): # j -> 1 dx/2
    ssm1 = start_point + stepping - 1
    for j in xrange(start_point, ssm1): # j -> 1 dx/2
        # Eq.(4.24), 'qn' = the next state of 'q', (u_m)nj -> qn, (u_m)(n-1/2)(j+-1/2) -> q 
        mtx_qn[:, j+1] = 0.5 * (mtx_q[:, j] + mtx_q[:, j+1] + mtx_s[:, j] - mtx_s[:, j+1]) 
        # Eq.(4.27) and Eq.(4.36), 'l' means '-' and 'r' means '+'.
        uxl = np.asarray((mtx_qn[:, j+1] - mtx_q[:, j] - 0.5 * dt * mtx_qt[:, j]) / (dx / 2.0))     
        uxr = np.asarray((mtx_q[:, j+1] + 0.5 * dt * mtx_qt[:, j+1] - mtx_qn[:, j+1]) / (dx / 2.0))
        # Eq.(4.38) and Eq.(4.39)
        mtx_qx[:, j+1] = np.asmatrix((uxl * (abs(uxr))**weight + uxr * (abs(uxl))**weight) \
                                      / ((abs(uxl))**weight + (abs(uxr))**weight + 10**(-60)))

    #for j in xrange(1, mtx_length):
    for j in xrange(start_point + 1, start_point + stepping):
        mtx_q[:, j] = mtx_qn[:, j]

    # IMPORTANT: mtx_q and mtx_qx have to be translated backward 1 dx per 1 dt (2 iterations)
    if i % 2 != 0:
        for j in xrange(1, mtx_length):
        #for j in xrange(start_point + 1, start_point + stepping):
            mtx_q[:, j-1] = mtx_q[:, j]
            mtx_qx[:, j-1] = mtx_qx[:, j]

        start_point -= 1
        stepping += 2

        # output region
        for j in xrange(0, mtx_length):
        #for j in xrange(start_point, start_point + stepping):
            status_rho[j] = mtx_q[0, j]
            status_vel[j] = mtx_q[1, j] / mtx_q[0, j]
            status_p[j] = (ga - 1.0) * (mtx_q[2, j] - 0.5 * mtx_q[0, j] * status_vel[j]**2)
    
        # making plots of gas status for different time intervals
        plt.subplot(311)
        plot_rho = plt.scatter(xx, status_rho, color="r")
        plt.xlabel("x")
        plt.ylabel("density")
        plt.xlim(-0.55, 0.55)
        plt.ylim(-0.1, 1.1)
        plt.subplot(312)
        plot_vel = plt.scatter(xx, status_vel, color="g")
        plt.xlim(-0.55, 0.55)
        plt.ylim(-0.1, 1.1)
        plt.xlabel("x")
        plt.ylabel("velocity")
        plt.subplot(313)
        plot_p = plt.scatter(xx, status_p, color="b")
        plt.xlim(-0.55, 0.55)
        plt.ylim(-0.1, 1.1)
        plt.xlabel("x")
        plt.ylabel("pressure")
        frame_seq.append((plot_rho, plot_vel, plot_p))

    # output text files which contain gas status of each point for different time intervals
    file = open("%03d" % (i + 1) + ".dat", 'w')
    for j in xrange(0, mtx_length):
        file.write(str(xx[j]) + " " + str(status_rho[j]) + " " + str(status_vel[j]) + " " + str(status_p[j]) + "\n")
        #print '{0:7f} {1:7f} {2:7f} {3:7f}'.format(xx[j], status_rho[j], status_vel[j], status_p[j])

    file.close()

ani = animation.ArtistAnimation(fig, frame_seq, interval=25, repeat_delay=300, blit=True)
ani.save('mySodTube.mp4', fps=10);

plt.show()

