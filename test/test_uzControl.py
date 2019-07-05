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
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

# from ConcentricTubeRobot.CTR_model import moving_CTR
from CTRmodel import moving_CTR
from CTRmodel import plot_3D
from TrajectoryGenerator import TrajectoryGenerator, TrajectoryRetreiver
from CurvatureController import Jacobian

Kp_Uz = np.eye(3) * 10


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


def main():
    """
        Calculates and compare forward kinematics from model directly and Jacobian technique.
    """

    a_ans = (2*np.pi)/3
    runtime = time.time()
    total_time = 5  # (seconds)
    dt = 0.1
    time_stamp = int(total_time/dt)
    t = dt
    i = 1
    jac_delta_uz = np.ones(3) * 1e-1

    model = lambda q, uz: moving_CTR(q, uz)
    q_static = np.array([0.001, 0.001, 0.001, 0.001, a_ans, a_ans])

    Uz_traj_pos = np.zeros((3, time_stamp))
    # Uz_end_model_pos = np.zeros((3, time_stamp))

    uz0_start = np.array([[0.0, 0.0, 0.0]]).transpose()

    uz0_cur_pos = np.zeros((3, time_stamp))
    Uz_end_cur_pos = np.zeros((3, time_stamp))

    Uz_traj_vel = np.zeros((3, time_stamp))
    delta_uz0 = np.zeros((3, time_stamp))

    delta_Uz = np.zeros((3, time_stamp))

    (r1,r2,r3,Uz_end) = moving_CTR(q_static, uz0_start)
    Uz_end_cur_pos[:, 0] = Uz_end.flatten()

    if quintic_fn:
        a1_coeffs = [[], [], []]
        a2_coeffs = [[], [], []]
        a3_coeffs = [[], [], []]
        waypoints = [Uz_end_cur_pos[:, 0], [0.0, 0.0, 0.0]]

        for x in range(len(waypoints)):
            traj = TrajectoryGenerator(waypoints[x], waypoints[(x + 1) % len(waypoints)], total_time)
            traj.solve()
            a1_coeffs[x] = traj.x_c
            a2_coeffs[x] = traj.y_c
            a3_coeffs[x] = traj.z_c

        quintic = TrajectoryRetreiver()

    while i <= time_stamp-1:
        # runtime = time.time()
        x_dim = np.zeros(3)  # just for size TODO: change to just integer
        if quintic_fn:
            Uz_traj_pos[0, i] = quintic.calculate_position(a1_coeffs[0], t)    # FOR INTEGRAL
            Uz_traj_pos[1, i] = quintic.calculate_position(a2_coeffs[0], t)
            Uz_traj_pos[2, i] = quintic.calculate_position(a3_coeffs[0], t)
        else:
            Uz_traj_pos[:, i] = alpha_position(t, total_time)

        if quintic_fn:
            Uz_traj_vel[0, i] = quintic.calculate_velocity(a1_coeffs[0], t)
            Uz_traj_vel[1, i] = quintic.calculate_velocity(a2_coeffs[0], t)
            Uz_traj_vel[2, i] = quintic.calculate_velocity(a3_coeffs[0], t)
        else:
            Uz_traj_vel[:, i] = alpha_velocity(t, total_time)

        delta_Uz[:, i] = Uz_traj_pos[:, i] - Uz_end_cur_pos[:, i-1]

        # get trajectory from Jacobian
        r_jac = Jacobian(jac_delta_uz, x_dim, q_static, uz0_cur_pos[:, i-1], model)
        J_inv = r_jac.p_inv()
        delta_uz0[:, i] = J_inv @ (Uz_traj_vel[:, i] + Kp_Uz @ delta_Uz[:, i])

        uz0_cur_pos[:, i] = uz0_cur_pos[:, i-1] + delta_uz0[:, i].copy() * dt
        # Uz_end_cur_pos += Uz_traj_vel[:, i].copy() * dt  # TODO: change to Forward Kinematics
        (r1,r2,r3,Uz) = model(q_static, uz0_cur_pos[:, i])
        Uz_end_cur_pos[:, i] = Uz.flatten()

        # print(i, time.time()-runtime)
        t += dt
        i += 1

    print("Done", time.time()-runtime)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    colors = cm.rainbow(np.linspace(0, 1, len(Uz_end_cur_pos.transpose())))
    for y, c in zip(Uz_end_cur_pos.transpose(), colors):
        # plt.scatter(x, y, color=c)
        ax.scatter(y[0], y[1], y[2], linewidth=1, color=c)
    ax.scatter(Uz_traj_pos[0], Uz_traj_pos[1], Uz_traj_pos[2], linewidth=1, label='Uz_traj_pos', marker='x')
    # ax.plot3D(Uz_end_cur_pos[0], Uz_end_cur_pos[1], Uz_end_cur_pos[2], linewidth=1, linestyle='--', label='Uz_end_jac')
    ax.scatter(Uz_end_cur_pos[0, -1], Uz_end_cur_pos[1, -1], Uz_end_cur_pos[2, -1], label=Uz_end_cur_pos[:, -1], marker='o')

    print('Final Uz:', Uz_end_cur_pos[:, -1])

    # (r1,r2,r3,Uz) = moving_CTR(q_ans, uz_0)
    # plot_3D(ax, r1, r2, r3, 'final position')
    ax.legend()
    ax.set_xlabel('tube1')
    ax.set_ylabel('tube2')
    ax.set_zlabel('tube3')
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
    plt.plot(tt, error_d(delta_Uz.transpose()), label='delta Uz')
    plt.title('uz0_cur_pos')
    plt.legend()

    # plt.subplots(1)
    # tt = np.arange(0.0, total_time, dt)
    # if quintic_fn:
    #     plt.plot(tt, quintic.calculate_position(a1_coeffs[0], tt))
    # else:
    #     plt.plot(tt, alpha_position(tt, total_time))
    # plt.title('uz0 pos trajectory')

    # plt.subplots(1)
    # if quintic_fn:
    #     plt.plot(tt, quintic.calculate_velocity(a1_coeffs[0], tt))
    # else:
    #     plt.plot(tt, alpha_velocity(tt, total_time))
    # plt.title('uz0 vel trajectory')

    plt.show()

def error_d(delta_x):
    distance = []
    for i in np.arange(len(delta_x)):
        distance.append(np.sqrt(delta_x[i, 0]**2+delta_x[i, 1]**2+delta_x[i, 2]**2))
    print(distance[-1])
    return distance

if __name__ == "__main__":
    quintic_fn = True
    main()
