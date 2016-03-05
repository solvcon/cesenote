#!/usr/bin/python

"""
Sod tube condition:
    iteration: 100
    x range: 102 points btw -0.505 ~ 0.505
    two region: left region, -0.505 ~ -0.005 (51 points)
                right region, 0.005 ~  0.505 (51 points)
"""

import matplotlib
matplotlib.use('TKAgg')

import my_cese_solver as mycese

import matplotlib.pyplot as plt
import matplotlib.animation as animation

ANIMATION_MODE = 1  # 0->w/o animation; 1->w/ animation
IT = 100
solver = mycese.CESESolver(IT, 1, 0.505, -0.505, 0.2)

fig = plt.figure()
frame_seq = []

for i in range(0, IT):
    solver.run_cese_iteration()
    solver.shift_cese_solution()
    solver.update_solution()
    #solver.export_solution(i)

    if not ANIMATION_MODE:
        continue

    x_axis = solver.get_current_frame()
    status_rho = solver.get_density()
    status_vel = solver.get_velocity()
    status_p = solver.get_pressure()

    if ANIMATION_MODE & i % 2 != 0:
        plt.subplot(311)
        plot_rho = plt.scatter(x_axis, status_rho, color="r")
        plt.xlabel("x")
        plt.ylabel("density")
        plt.xlim(-0.55, 0.55)
        plt.ylim(-0.1, 1.1)
        plt.subplot(312)
        plot_vel = plt.scatter(x_axis, status_vel, color="g")
        plt.xlim(-0.55, 0.55)
        plt.ylim(-0.1, 1.1)
        plt.xlabel("x")
        plt.ylabel("velocity")
        plt.subplot(313)
        plot_p = plt.scatter(x_axis, status_p, color="b")
        plt.xlim(-0.55, 0.55)
        plt.ylim(-0.1, 1.1)
        plt.xlabel("x")
        plt.ylabel("pressure")
        frame_seq.append((plot_rho, plot_vel, plot_p))

if ANIMATION_MODE:
    ani = animation.ArtistAnimation(fig, frame_seq, interval=25, repeat_delay=300, blit=True)
    ani.save('mySodTube.mp4', fps=10);

    plt.show()
