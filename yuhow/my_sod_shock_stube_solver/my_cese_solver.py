"""
Sod Shock Tube Problem -- CESE solver v3
Author: You-Hao Chang

Journal of Computational Physics 119, 295-324 (1995)
The Method of Space-Time Conservation Element and Solution
Element -- A New Approach for Solving the Navier-Stokes and 
Eluer Equations

"""

import numpy as np

# global variable
GAMMA = 1.4         # gamma = Cp/Cv

RHO_L = 1.0
P_L   = 1.0
V_L   = 0.0
RHO_R = 0.125
P_R   = 0.1
V_R   = 0.0

class Data(object):
    """
    CESE data container of shock tube 
    """
    _data_content = ['iteration',  # number of iteration
                     'npx',        # number of mesh point on x-axix
                     'space_size', # dx/2, mesh size of space 
                     'time_size',  # dt/2, mesh size of time
                     'mtx_f',      # matrix of fm,k
                     'mtx_q',      # matrix of current state (u_m)nj 
                     'mtx_qn',     # matrix of next state (u_m)nj
                     'mtx_qx',     # matrix of (u_kx)nj
                     'mtx_qt',     # matrix of (u_mt)nj
                     'mtx_s',      # matrix of (s_m)nj
                     'uxl',        # matrix of (u_mx-)nj
                     'uxr',        # matrix of (u_mx+)nj
                     'weight',     # weighting factor
                     'start_x',    # start point of the region we focus on
                     'end_x',      # end point of the region we focus on
                     'current_t',  # current time (start from t=0),
                     'target_t',   # how long we want this shock tube evolve
                     'x_axis',     # x-axis
                     'solution'    # solution (density, velocity, pressure)
                    ]

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if k in self._data_content:
                self.__dict__[k] = v
        

class CESESolver(object):
    """
    CESE solver for one dimensional shock tube problem
    """
    def __init__(self,
                 iteration,
                 weight,
                 start_x,
                 end_x,
                 target_t):
        
        global RHO_L, RHO_R, V_L, V_R, P_L, P_R

        npx        = iteration + 2
        space_size = (start_x - end_x) / (npx - 1) / 2
        time_size  = (target_t - 0) / iteration 
        
        mtx_f  = np.asmatrix(np.zeros(shape=(3, 3)))
        mtx_q  = np.asmatrix(np.zeros(shape=(3, npx)))
        mtx_qn = np.asmatrix(np.zeros(shape=(3, npx)))
        mtx_qx = np.asmatrix(np.zeros(shape=(3, npx)))
        mtx_qt = np.asmatrix(np.zeros(shape=(3, npx)))
        mtx_s  = np.asmatrix(np.zeros(shape=(3, npx)))
        uxl = np.zeros(shape=(3,1))
        uxr = np.zeros(shape=(3,1))

        for j in xrange(0, npx):
            # Eq (4.1), u1, u2 and u3
            if j < npx / 2:
                mtx_q[0, j] = RHO_L
                mtx_q[1, j] = RHO_L * V_L
                mtx_q[2, j] = P_L / (GAMMA - 1.0) + 0.5 * RHO_L * V_L**2
            else:
                mtx_q[0, j] = RHO_R
                mtx_q[1, j] = RHO_R * V_R
                mtx_q[2, j] = P_R / (GAMMA - 1.0) + 0.5 * RHO_R * V_R**2

        current_t = 0.

        x_axis  = [0. for idx in range(npx)] # x-axis
        x_axis[0] = -1 * space_size * float(iteration + 1)
        for j in xrange(0, npx - 1):
            x_axis[j+1] = x_axis[j] + (space_size * 2)

        status_rho  = [RHO_R if idx >= npx / 2 else RHO_L for idx in range(npx)] # density
        status_vel  = [V_R if idx >= npx / 2 else V_L for idx in range(npx)] # velocity
        status_p    = [P_R if idx >= npx / 2 else P_L for idx in range(npx)] # pressure

        solution = [x_axis, status_rho, status_vel, status_p]

        self._data = Data(iteration  = iteration,
                          npx        = npx, 
                          space_size = space_size, 
                          time_size  = time_size,  
                          mtx_f      = mtx_f,      
                          mtx_q      = mtx_q,      
                          mtx_qn     = mtx_qn,     
                          mtx_qx     = mtx_qx,     
                          mtx_qt     = mtx_qt,     
                          mtx_s      = mtx_s,      
                          uxl        = uxl,        
                          uxr        = uxr,        
                          weight     = weight,
                          start_x    = start_x,    
                          end_x      = end_x,      
                          current_t  = current_t,  
                          target_t   = target_t,
                          x_axis     = x_axis,
                          solution   = solution)

    @property
    def data(self):
        return self._data

    def run_cese_iteration(self):
        """
        evaluate the gas status based on CESE method as time goes on
        two steps: 1) fill mtx_f, mtx_qt and mtx_s according to the current
                      status of gas
                   2) using the matrices of step 1 to extract the status of 
                      gas after dt/2 for each iteration
        """
        data = self._data
        mtx_length = data.npx

        space_size = data.space_size
        time_size  = data.time_size
        mtx_q = data.mtx_q
        mtx_qn = data.mtx_qn
        mtx_f = data.mtx_f
        mtx_qt = data.mtx_qt
        mtx_qx = data.mtx_qx
        mtx_s = data.mtx_s
        
        uxl = data.uxl
        uxr = data.uxr

        weight = data.weight

        # step 1: evaluate the current status of gas
        for j in xrange(0, mtx_length):
        
            # Eq. (4.7): constructing the matrix fm,k 
            # other reference: Yung-Yu's notes -> Sec.3.1, Eq. (3.14)
            w2 = mtx_q[1, j] / mtx_q[0, j]  # u2/u1
            w3 = mtx_q[2, j] / mtx_q[0, j]  # u3/u1
            mtx_f[0, 0] = 0.0
            mtx_f[0, 1] = 1.0
            mtx_f[0, 2] = 0.0
            mtx_f[1, 0] = -0.5 * (3.0 - GAMMA) * w2**2
            mtx_f[1, 1] = (3.0 - GAMMA) * w2
            mtx_f[1, 2] = GAMMA - 1.0
            mtx_f[2, 0] = (GAMMA - 1.0) * w2**3 - GAMMA * w2 * w3
            mtx_f[2, 1] = GAMMA * w3 - 1.5 * (GAMMA - 1.0) * w2**2 
            mtx_f[2, 2] = GAMMA * w2
        
            # Eq.(4.17), (u_mt)nj = -(f_mx)nj = -(fm,k*u_kx)nj, (u_mt)nj -> qt, (u_kx)nj -> qx
            mtx_qt[:, j] = -1.0 * mtx_f * mtx_qx[:, j]
        
            # Eq.(4.25), (u_m)nj -> q, (u_mt)nj -> qt, (u_kx)nj -> qx
            # for s0: u_0x -> mtx_qx[0, j], f_0 -> mtx_q[1, j], f_0t -> f0,k*u_kt 
            #     s1: u_1x -> mtx_qx[1, j], f_1 -> mtx_f[1, 2]*mtx_q[2, j]+mtx_f[1, 0]*mtx_q[0, j]+mtx_f[1, 1]*mtx_q[1, j]
            #                               f_1t-> mtx_f[1, 0]*mtx_qx[0, j]+mtx_f[1, 1]*mtx_qx[1, j]+mtx_f[1, 2]*mtx_qx[2, j]
            #     s2: u_2x -> mtx_qx[2, j], f_2 -> mtx_f[2, 0]*mtx_q[0, j]+mtx_f[2, 1]*mtx_q[1, j]+mtx_f[2, 2]*mtx_q[2, j]
            #                               f_2t-> *mtx_qx[0, j]+f32*mtx_qx[1, j]+f33*mtx_qx[2, j]
            mtx_s[:, j] = 0.5 * space_size * mtx_qx[:, j] + (time_size / space_size) * mtx_f * mtx_q[:, j] - 0.5 * time_size * (time_size / space_size) * mtx_f * mtx_f * mtx_qx[:, j]

        # step 2: evaluate the status of gas after time stamp moves forward by 1 dt/2
        mm = mtx_length - 1
        for j in xrange(0, mm): # j -> 1 dx/2
            # Eq.(4.24), 'qn' = the next state of 'q', (u_m)nj -> qn, (u_m)(n-1/2)(j+-1/2) -> q 
            mtx_qn[:, j+1] = 0.5 * (mtx_q[:, j] + mtx_q[:, j+1] + mtx_s[:, j] - mtx_s[:, j+1]) 
            # Eq.(4.27) and Eq.(4.36), 'l' means '-' and 'r' means '+'.
            uxl = np.asarray((mtx_qn[:, j+1] - mtx_q[:, j] - time_size * mtx_qt[:, j]) / space_size)     
            uxr = np.asarray((mtx_q[:, j+1] + time_size * mtx_qt[:, j+1] - mtx_qn[:, j+1]) / space_size)
            # Eq.(4.38) and Eq.(4.39)
            mtx_qx[:, j+1] = np.asmatrix((uxl * (abs(uxr))**weight + uxr * (abs(uxl))**weight) \
                                          / ((abs(uxl))**weight + (abs(uxr))**weight + 10**(-60)))
        
        for j in xrange(1, mtx_length):
            mtx_q[:, j] = mtx_qn[:, j]

        data.current_t += time_size

    def shift_cese_solution(self):
        """
        shifting mtx_q and mtx_qx:
        mtx_q and mtx_qx have to be translated backward 1 dx per 1 dt (2 iterations)
        due to our implementation
        """
        data = self._data
        mtx_length = data.npx
        time_size  = data.time_size
        mtx_q = data.mtx_q
        mtx_qx = data.mtx_qx

        current_t = data.current_t

        if int(current_t / time_size) % 2 == 0:
            for j in xrange(1, mtx_length):
                mtx_q[:, j-1] = mtx_q[:, j]
                mtx_qx[:, j-1] = mtx_qx[:, j]

    def update_solution(self):
        global GAMMA

        data = self._data
        mtx_length = data.npx
        time_size  = data.time_size
        mtx_q = data.mtx_q
        current_t = data.current_t
        solution = data.solution

        if int(current_t / time_size) % 2 == 0:
            for j in xrange(0, mtx_length):
                solution[1][j] = mtx_q[0, j]
                solution[2][j] = mtx_q[1, j] / mtx_q[0, j]
                solution[3][j] = (GAMMA - 1.0) * (mtx_q[2, j] - 0.5 * mtx_q[0, j] * solution[2][j]**2)

    def get_density(self):
        return self._data.solution[1]

    def get_velocity(self):
        return self._data.solution[2]

    def get_pressure(self):
        return self._data.solution[3]

    def get_current_frame(self):
        return self._data.x_axis

    def get_full_solution(self):
        return self._data.solution

    def export_solution(self, it):
        """
        export text files of solution for each iteration
        format: position density velocity pressure
        """
        data = self._data
        mtx_length = data.npx
        solution = data.solution

        file = open("%03d" % (it + 1) + ".dat", 'w')
        for j in xrange(0, mtx_length):
            file.write(str(solution[0][j]) + " " + str(solution[1][j]) + " " + str(solution[2][j]) + " " + str(solution[3][j]) + "\n")
            #print '{0:7f} {1:7f} {2:7f} {3:7f}'.format(x_axis[j], solution[0][j], solution[1][j], solution[2][j])

        file.close()

