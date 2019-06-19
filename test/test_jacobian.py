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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# from ConcentricTubeRobot.CTR_model import moving_CTR
from test_model import moving_CTR
from test_model import plot_3D
from TrajectoryGenerator import TrajectoryGenerator
from controller import Jacobian


def alpha_position(t, total_time):
    """
        Calculates a position given current time.

        Args
            t: Time at which to calculate the position

        Returns
            Position
    """
    return (2 * np.pi / 3 * total_time) * t


def alpha_velocity(t, total_time):
    """
        Calculates a velocity given current time.

        Args
            t: Time at which to calculate the velocity

        Returns
            Velocity
    """
    return (2 * np.pi / 3 * total_time)


def main():
    """
    Calculates and compare forward kinematics from model directly and Jacobian technique.
    """

    total_time = 10  # (seconds)
    dt = 0.1
    time_stamp = int(total_time/dt)
    t = dt
    i = 0

    uz_0 = np.array([[0, 0, 0]]).transpose()
    model = lambda q, uz_0: moving_CTR(q, uz_0)

    q_model_pos = np.zeros((6, time_stamp))  # [BBBaaa]
    x_model_pos = np.zeros((3, time_stamp))  # [r]

    a_ans = 2 * np.pi / 3
    q_start = np.array([0, 0, 0, 0, 0, 0])
    q_ans = np.array([0, 0, 0, a_ans, a_ans, a_ans])

    if jac_test:
        (r1,r2,r3,Uz) = moving_CTR(q_start, uz_0)
        x_jac_cur_pos = r1[-1]
        q_jac_cur_pos = 0
        
        q_jac_pos = np.zeros((6, time_stamp))  # [BBBaaa]
        x_jac_pos = np.zeros((3, time_stamp))  # [r]

        q_vel = np.zeros((6, time_stamp))  # [BBBaaa]
        x_vel = np.zeros((3, time_stamp))  # [r]

    while i <= time_stamp-1:
        runtime = time.time()
        delta_q = np.ones(6) * 1e-3
        x = np.zeros(3)  # just for size TODO: change to just integer
        q_model_pos[3:6, i] = alpha_position(t, total_time)
        # print('t:', t)
        # print('i:', i)
        # print(alpha_position(t, total_time))

        # get trajectory derectly from dynamics model
        (r1,r2,r3,Uz) = moving_CTR(q_model_pos[:, i].flatten(), uz_0)

        x_model_pos[:, i] = np.array(r1[-1])

        if jac_test:
            q_vel[3:6, i] = alpha_velocity(t, total_time)
            
            # get trajectory from Jacobian
            r_jac = Jacobian(delta_q, x, q_vel[:, i].flatten(), uz_0, model)
            r_jac.jac_approx()
            J = r_jac.J
            x_vel[:, i] = J @ q_vel[:, i]

            x_jac_cur_pos += x_vel[:, i].copy() * dt
            q_jac_cur_pos += q_vel[:, i].copy() * dt  # TODO: is it += or just + ???
            x_jac_pos[:, i] = x_jac_cur_pos
            q_jac_pos[:, i] = q_jac_cur_pos  # TODO: is it += or just + ???

        #     print(x_vel[:, i].copy() * dt)
        #     print(x_jac_cur_pos)

        # print(i, time.time()-runtime)
        # print('\nx_model_pos\n', x_model_pos[:, i])
        # print('\nx_jac_pos\n', x_jac_pos[:, i])
        t += dt
        i += 1

    print("Done")
    # print('\nq_vel\n', q_vel)
    # print('\nx_vel\n', x_vel)
    # print('\nq_model_pos\n', q_model_pos)
    # print('\nx_model_pos\n', x_model_pos)
    # print('\nq_jac_pos\n', q_jac_pos)
    # print('\nx_jac_pos\n', x_jac_pos)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(x_model_pos[0], x_model_pos[1], x_model_pos[2], linewidth=1, linestyle=':', label='x_model_pos')
    if jac_test:
        ax.plot3D(x_jac_pos[0], x_jac_pos[1], x_jac_pos[2], linewidth=1, linestyle='--', label='x_jac_pos')

    (r1,r2,r3,Uz) = moving_CTR(q_ans, uz_0)
    plot_3D(ax, r1, r2, r3, 'final position')

    ax.legend()
    plt.show()

if __name__ == "__main__":
    jac_test = True
    main()
