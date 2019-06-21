"""
Simulate a CTR following a 3D trajectory

Author: Izzat Kamarudzaman

Adapted from code by Python Robotics, Daniel Ingram (daniel-s-ingram)
"""

import numpy as np
import time
import sys
sys.path.append("../")
sys.path.append("./ConcentricTubeRobot/")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# from ConcentricTubeRobot.CTR_model import moving_CTR
from test_model import moving_CTR
from test_model import plot_3D
from TrajectoryGenerator import TrajectoryGenerator
from CurvatureController import Jacobian


def alpha_position(t, total_time):
    """
        Calculates a position given current time.

        Args
            t: Time at which to calculate the position

        Returns
            Position
    """
    return np.array(((2 * np.pi) / (3 * total_time)) * t)


def alpha_velocity(t, total_time):
    """
        Calculates a velocity given current time.

        Args
            t: Time at which to calculate the velocity

        Returns
            Velocity
    """
    return np.array(((2 * np.pi) / (3 * total_time)) -t+t )


def calculate_position(c, t):
    """
        Calculates a position given a set of quintic coefficients and a time.

        Args
            c: List of coefficients generated by a quintic polynomial 
                trajectory generator.
            t: Time at which to calculate the position

        Returns
            Position
    """
    return c[0] * t**5 + c[1] * t**4 + c[2] * t**3 + c[3] * t**2 + c[4] * t + c[5]


def calculate_velocity(c, t):
    """
        Calculates a velocity given a set of quintic coefficients and a time.

        Args
            c: List of coefficients generated by a quintic polynomial 
                trajectory generator.
            t: Time at which to calculate the velocity

        Returns
            Velocity
    """
    return 5 * c[0] * t**4 + 4 * c[1] * t**3 + 3 * c[2] * t**2 + 2 * c[3] * t + c[4]


def calculate_acceleration(c, t):
    """
        Calculates an acceleration given a set of quintic coefficients and a time.

        Args
            c: List of coefficients generated by a quintic polynomial 
                trajectory generator.
            t: Time at which to calculate the acceleration

        Returns
            Acceleration
    """
    return 20 * c[0] * t**3 + 12 * c[1] * t**2 + 6 * c[2] * t + 2 * c[3]


def main():
    """
        Calculates and compare forward kinematics from model directly and Jacobian technique.
    """

    runtime = time.time()
    total_time = 5  # (seconds)
    dt = 0.1
    time_stamp = int(total_time/dt)
    t = dt
    i = 0
    delta_uz = np.ones(3) * 1e-1

    model = lambda q, uz: moving_CTR(q, uz)
    q_static = np.array([0, 0, 0, 0, 0, 0])

    uz0_model_pos = np.zeros((3, time_stamp))  # [BBBaaa]
    Uz_end_model_pos = np.zeros((3, time_stamp))  # [r]

    U_end_1 = U_end_2 = U_end_3 = 0.1
    uz0_start = np.array([[0.0, 0.0, 0.0]]).transpose()
    uz0_end = np.array([[U_end_1, U_end_2, U_end_3]]).transpose()

    if jac_test:
        (r1,r2,r3,Uz_end) = moving_CTR(q_static, uz0_start)
        Uz_end_jac_cur = Uz_end.flatten()
        # print('First x_jac_cur_pos:', x_jac_cur_pos)
        uz0_jac_cur_pos = 0
        
        uz0_jac_pos = np.zeros((3, time_stamp))
        Uz_end_jac = np.zeros((3, time_stamp))

        uz0_vel = np.zeros((3, time_stamp))
        Uz_end_x_vel = np.zeros((3, time_stamp))

    if quintic:
        a1_coeffs = [[], [], []]
        a2_coeffs = [[], [], []]
        a3_coeffs = [[], [], []]
        #  all B (0 -> 0), all alpha (0 -> 2pi/3) 
        waypoints = [[0.0, 0.0, 0.0], [0.03, 0.07, 0.02], [U_end_1, U_end_2, U_end_3]]

        for x in range(len(waypoints)):
            traj = TrajectoryGenerator(waypoints[i], waypoints[(i + 1) % len(waypoints)], total_time)
            traj.solve()
            a1_coeffs[x] = traj.x_c
            a2_coeffs[x] = traj.y_c
            a3_coeffs[x] = traj.z_c

        # print('a1_coeffs:\n', a1_coeffs)
        # print('a2_coeffs:\n', a2_coeffs)
        # print('a3_coeffs:\n', a3_coeffs)

    while i <= time_stamp-1:
        # runtime = time.time()
        x = np.zeros(3)  # just for size TODO: change to just integer
        if quintic:
            uz0_model_pos[0, i] = calculate_position(a1_coeffs[0], t)
            uz0_model_pos[1, i] = calculate_position(a2_coeffs[0], t)
            uz0_model_pos[2, i] = calculate_position(a3_coeffs[0], t)
        else:
            uz0_model_pos[:, i] = alpha_position(t, total_time)
        # print('t:', t)
        # print('i:', i)
        # print(alpha_position(t, total_time))

        # get trajectory derectly from dynamics model
        (r1,r2,r3,Uz) = moving_CTR(q_static, uz0_model_pos[:, i])

        Uz_end_model_pos[:, i] = Uz.flatten()

        if jac_test:
            if quintic:
                uz0_vel[0, i] = calculate_velocity(a1_coeffs[0], t)
                uz0_vel[1, i] = calculate_velocity(a2_coeffs[0], t)
                uz0_vel[2, i] = calculate_velocity(a3_coeffs[0], t)
            else:
                uz0_vel[:, i] = alpha_velocity(t, total_time)
            
            # get trajectory from Jacobian
            r_jac = Jacobian(delta_uz, x, q_static, uz0_model_pos[:, i], model)
            r_jac.jac_approx()
            J = r_jac.J
            Uz_end_x_vel[:, i] = J @ uz0_vel[:, i]

            Uz_end_jac_cur += Uz_end_x_vel[:, i].copy() * dt
            uz0_jac_cur_pos += uz0_vel[:, i].copy() * dt  # TODO: is it += or just + ???
            Uz_end_jac[:, i] = Uz_end_jac_cur
            uz0_jac_pos[:, i] = uz0_jac_cur_pos  # TODO: is it += or just + ???

        #     print(x_vel[:, i].copy() * dt)
            # print(x_jac_cur_pos)

        # print(i, time.time()-runtime)
        # print('\nx_model_pos\n', x_model_pos[:, i])
        # print('\nx_jac_pos\n', x_jac_pos[:, i])
        t += dt
        i += 1

    print("Done", time.time()-runtime)
    # print('\nq_vel\n', q_vel)
    # print('\nx_vel\n', x_vel)
    # print('\nq_model_pos\n', q_model_pos)
    # print('\nx_model_pos\n', x_model_pos)
    # print('\nq_jac_pos\n', q_jac_pos)
    # print('\nx_jac_pos\n', x_jac_pos)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(Uz_end_model_pos[0], Uz_end_model_pos[1], Uz_end_model_pos[2], linewidth=1, label='Uz_end_model_pos')
    if jac_test:
        ax.plot3D(Uz_end_jac[0], Uz_end_jac[1], Uz_end_jac[2], linewidth=1, linestyle='--', label='Uz_end_jac')

    # (r1,r2,r3,Uz) = moving_CTR(q_ans, uz_0)
    # plot_3D(ax, r1, r2, r3, 'final position')
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


    plt.subplots(1)
    tt = np.arange(0.0, total_time, dt)
    if quintic:
        plt.plot(tt, calculate_position(a1_coeffs[0], tt))
    else:
        plt.plot(tt, alpha_position(tt, total_time))
    plt.title('Tube Rotation (rad)')

    plt.subplots(1)
    if quintic:
        plt.plot(tt, calculate_velocity(a1_coeffs[0], tt))
    else:
        plt.plot(tt, alpha_velocity(tt, total_time))
    plt.title('Rotation Velocity (rad/s)')

    plt.show()

if __name__ == "__main__":
    jac_test = True
    quintic = True
    main()