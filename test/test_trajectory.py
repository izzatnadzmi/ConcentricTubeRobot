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
from mpl_toolkits.mplot3d import Axes3D
# from ConcentricTubeRobot.CTR_model import moving_CTR
from test_model import moving_CTR
from TrajectoryGenerator import TrajectoryGenerator, TrajectoryRetreiver
from controller import Jacobian


show_animation = True

def CTR_sim(a1_c, a2_c, a3_c, total_time):
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
    i = 0
    delta_q = np.ones(6) * 1e-1

    uz_0 = np.array([[0, 0, 0]]).transpose()
    model = lambda q, uz_0: moving_CTR(q, uz_0)

    # q_model_pos = np.zeros((6, time_stamp))  # [BBBaaa]
    x_model_pos = np.zeros((3, time_stamp))  # [r]

    # x_jac_cur_pos = r1[-1]
    # print('First x_jac_cur_pos:', x_jac_cur_pos)
    q_jac_cur_pos = 0
    
    q_jac_pos = np.zeros((6, time_stamp))  # [BBBaaa]
    x_jac_pos = np.zeros((3, time_stamp))  # [r]

    q_vel = np.zeros((6, time_stamp))  # [BBBaaa]
    x_vel = np.zeros((3, time_stamp))  # [r]

    quintic = TrajectoryRetreiver()

    while i <= time_stamp-1:
        # runtime = time.time()
        x = np.zeros(3)  # just for size TODO: change to just integer
        q_model_pos[3, i] = quintic.calculate_position(a1_c[0], t)
        q_model_pos[4, i] = quintic.calculate_position(a2_c[0], t)
        q_model_pos[5, i] = quintic.calculate_position(a3_c[0], t)
        # print('t:', t)
        # print('i:', i)
        # print(alpha_position(t, total_time))

        # get trajectory directly from dynamics model
        # (r1,r2,r3,Uz) = moving_CTR(q_model_pos[:, i].flatten(), uz_0)

        # x_model_pos[:, i] = np.array(r1[-1])

        q_vel[3, i] = quintic.calculate_velocity(a1_c[0], t)
        q_vel[4, i] = quintic.calculate_velocity(a2_c[0], t)
        q_vel[5, i] = quintic.calculate_velocity(a3_c[0], t)

        # get trajectory from Jacobian
        r_jac = Jacobian(delta_q, x, q_model_pos[:, i].flatten(), uz_0, model)
        r_jac.jac_approx()
        J = r_jac.J
        x_vel[:, i] = J @ q_vel[:, i]

        x_jac_cur_pos += x_vel[:, i].copy() * dt
        q_jac_cur_pos += q_vel[:, i].copy() * dt  # TODO: is it += or just + ???
        x_jac_pos[:, i] = x_jac_cur_pos
        q_jac_pos[:, i] = q_jac_cur_pos  # TODO: is it += or just + ???

        #     print(x_vel[:, i].copy() * dt)
            # print(x_jac_cur_pos)

        # print(i, time.time()-runtime)
        # print('\nx_model_pos\n', x_model_pos[:, i])
        # print('\nx_jac_pos\n', x_jac_pos[:, i])
        t += dt
        i += 1

    print("Done", time.time()-runtime)


def main():
    """
    Calculates the x, y, z coefficients for the four segments 
    of the trajectory
    """
    #  all B (0 -> 0), all alpha (0 -> 2pi/3)
    a_ans = (2*np.pi)/3
    q_start = np.array([0, 0, 0, 0, 0, 0])
    q_end = np.array([0, 0, 0, a_ans, a_ans, a_ans])

    (r1,r2,r3,Uz) = moving_CTR(q_start, uz_0)
    x_jac_cur_pos = r1[-1]
    # print('First x_jac_cur_pos:', x_jac_cur_pos)


    waypoints = [[0.0, 0.0, 0.0], [a_ans, a_ans, a_ans]]
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

    CTR_sim(a1_coeffs, a2_coeffs, a3_coeffs, total_time)


if __name__ == "__main__":
    main()
