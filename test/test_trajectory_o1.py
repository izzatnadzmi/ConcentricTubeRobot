"""
Simulate a CTR following a 3D trajectory

Author: Izzat Kamarudzaman

Adapted from code by Python Robotics, Daniel Ingram (daniel-s-ingram)
"""

from math import cos, sin
import numpy as np
import time
import sys
sys.path.append("../")
sys.path.append("./ConcentricTubeRobot/")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
# from ConcentricTubeRobot.CTR_model import moving_CTR
from CTRmodel import moving_CTR, plot_3D
from TrajectoryGenerator import TrajectoryGenerator, TrajectoryRetreiver
from controller import Jacobian


show_animation = True

def CTR_sim(a1_c, a2_c, a3_c, total_time, q_start, q_end):
    """
    Calculates the necessary thrust and torques for the quadrotor to
    follow the trajectory described by the sets of coefficients
    x_c, y_c, and z_c.
    """
    runtime = time.time()
    total_time = 5  # (seconds)
    dt = 0.1
    time_stamp = int(total_time/dt)
    t = dt
    i = 1
    jac_del_q = np.ones(6) * 1e-1
    uz_0 = np.array([[0, 0, 0]]).transpose()

    model = lambda q, uz_0: moving_CTR(q, uz_0)

    q_des_pos = np.zeros((6, time_stamp))  # [BBBaaa]
    x_des_pos = np.zeros((3, time_stamp))  # [r]
    x_cur_pos = np.zeros((3, time_stamp))  # [r]

    delta_q = np.zeros(6)  # [BBBaaa]
    delta_x = np.zeros(3)  # [r]

    quintic = TrajectoryRetreiver()
    q_des_pos[:, 0] = q_start


    while i < time_stamp:
        # runtime = time.time()

        x = np.zeros(3)  # just for size TODO: change to just integer
        x_des_pos[0, i] = quintic.calculate_position(a1_c[0], t)
        x_des_pos[1, i] = quintic.calculate_position(a2_c[0], t)
        x_des_pos[2, i] = quintic.calculate_position(a3_c[0], t)
        # print('t:', t)
        # print('i:', i)
        # print(alpha_position(t, total_time))

        delta_x = x_des_pos[:, i] - x_cur_pos[:, i-1]
        # print('delta_x', delta_x)

        # get trajectory from Jacobian
        r_jac = Jacobian(jac_del_q, x, q_des_pos[:, i-1].flatten(), uz_0, model)
        J_inv = r_jac.p_inv()
        delta_q = J_inv @ delta_x
        # print('delta_q', delta_q)

        q_des_pos[:, i] = q_des_pos[:, i-1] + delta_q * dt
        (r1,r2,r3,Uz) = model(q_des_pos[:, i], uz_0)        # FORWARD KINEMATICS
        x_cur_pos[:, i] = r1[-1]
        # print(i, time.time()-runtime)
        t += dt
        i += 1

    print("Done", time.time()-runtime)
    print('x_des_pos[:, 1]:', x_des_pos[:, 1])
    print('x_des_pos[:, -1]:', x_des_pos[:, -1])
    print('q_des_pos[:, 0]:', q_des_pos[:, 0])
    print('q_des_pos[:, -1]:', q_des_pos[:, -1])

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    colors = cm.rainbow(np.linspace(0, 1, len(x_cur_pos.transpose())))
    for y, c in zip(x_cur_pos.transpose(), colors):
        # plt.scatter(x, y, color=c)
        ax.scatter(y[0], y[1], y[2], linewidth=1, color=c)
        # ax.plot3D(x_des_pos[0], x_des_pos[1], x_des_pos[2], linewidth=1, label='x_des_pos')
    ax.scatter(x_des_pos[0], x_des_pos[1], x_des_pos[2], linewidth=1, label='x_des_pos', marker='.')

    (r1,r2,r3,Uz) = moving_CTR(q_des_pos[:, 1], uz_0)
    plot_3D(ax, r1, r2, r3, 'start position')
    (r1,r2,r3,Uz) = moving_CTR(q_des_pos[:, -1], uz_0)
    plot_3D(ax, r1, r2, r3, 'final position')
    ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z ')
    # plt.axis('equal')
    # ax.set_aspect('equal')

    # # Create cubic bounding box to simulate equal aspect ratio
    # max_range = 0.2  # np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
    # Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(0)  # X.max()+X.min())
    # Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(0)  # Y.max()+Y.min())
    # Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(0.3)  # Z.max()+Z.min())
    # # Comment or uncomment following both lines to test the fake bounding box:
    # for xb, yb, zb in zip(Xb, Yb, Zb):
    #     ax.plot([xb], [yb], [zb], 'w')

    plt.show()


def main():
    """
    Calculates the x, y, z coefficients for the four segments 
    of the trajectory
    """
    #  all B (0 -> 0), all alpha (0 -> 2pi/3)
    a_ans = (2*np.pi)/3
    q_start = np.array([0.0001, 0.0001, 0.0001, a_ans, a_ans, a_ans])
    q_end = np.array([0.0001, 0.0001, 0.0001, a_ans + 0.2, a_ans + 0.2, a_ans + 0.2])
    uz_0 = np.array([[0, 0, 0]]).transpose()

    (r1,r2,r3,Uz) = moving_CTR(q_start, uz_0)
    x_cur_pos = r1[-1]
    (r1e,r2e,r3e,Uze) = moving_CTR(q_end, uz_0)
    x_end_pos = r1e[-1]

    # waypoints = [[0.0, 0.0, 0.0], [a_ans, a_ans, a_ans]]
    waypoints = [x_cur_pos, x_end_pos]
    a1_coeffs = []
    a2_coeffs = []
    a3_coeffs = []
    total_time = 5

    for x in range(len(waypoints)):
        traj = TrajectoryGenerator(waypoints[x], waypoints[(x + 1) % len(waypoints)], total_time)
        traj.solve()
        a1_coeffs.append(traj.x_c)
        a2_coeffs.append(traj.y_c)
        a3_coeffs.append(traj.z_c)

    print('START x_cur_pos:', x_cur_pos)
    print('END x_end_pos:', x_end_pos)
    CTR_sim(a1_coeffs, a2_coeffs, a3_coeffs, total_time, q_start, q_end)
    print('START q_start:', q_start)
    print('END q_end:', q_end)


if __name__ == "__main__":
    main()
