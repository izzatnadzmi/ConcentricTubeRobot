"""
Inverse kinematics for CTR using the Jacobian inverse method and movement test

Author: Izzat Kamarudzaman
Adapted from code by Python Robotics
                        Daniel Ingram (daniel-s-ingram) & Atsushi Sakai (@Atsushi_twi)
"""

from math import cos, sin
import numpy as np
import sys
from mpl_toolkits.mplot3d import Axes3D
sys.path.append("../")
# from CTR_model import moving_CTR
from test_model import moving_CTR
from TrajectoryGenerator import TrajectoryGenerator
from controller import Jacobian


# Simulation parameters
Kp = 2
dt = 0.1
N_ITERATIONS = 1000

# States
WAIT_FOR_NEW_GOAL = 1
MOVING_TO_GOAL = 2

show_animation = True

# Jacobian calculation parameters
jac_delta_q = np.ones(6) * 1e-2
uz_0 = np.array([[0, 0, 0]]).transpose()
model = lambda q, uz_0: moving_CTR(q, uz_0)


def main():  # pragma: no cover
    """
    Creates an arm using the NLinkArm class and uses its inverse kinematics
    to move it to the desired position.
    """

    q_now = np.array([0, 0, 0, 0, 0, 0])   # to np.array([0, 0, 0, np.pi*2/3, np.pi*2/3, np.pi*2/3])
    goal_pos = [0.015457, 0.057684, 0.269762]   # from [0.042224, -0.042224, 0.269772] -> (0, 0, 0)
    # arm = NLinkArm(link_lengths, q_now, goal_pos, show_animation)
    state = WAIT_FOR_NEW_GOAL
    solution_found = False

    while True:
        old_goal = np.array(goal_pos)
        goal_pos = np.array(goal_pos)
        x_now = forward_kinematics(q_now)
        errors, distance = distance_to_goal(x_now, goal_pos)
        print("distance", distance)

        # State machine to allow changing of goal before current goal has been reached
        if state is WAIT_FOR_NEW_GOAL:

            if distance > 0.1 and not solution_found:
                q_new, solution_found = inverse_kinematics(q_now, goal_pos)
                if not solution_found:
                    print("Solution could not be found.")
                    break
                elif solution_found:
                    state = MOVING_TO_GOAL
        elif state is MOVING_TO_GOAL:
            if distance > 0.1 and all(old_goal == goal_pos):
                q_now = q_new  # q_now + Kp * \
                    # ang_diff(q_new, q_now) * dt
            else:
                print("Done")
                # state = WAIT_FOR_NEW_GOAL   # TODO: need this?
                # solution_found = False
                break

        # arm.update_joints(q_now)  # TODO: plot scatter
        print("FINAL Q: ", q_now)


def inverse_kinematics(q_now, goal_pos):
    """
    Calculates the inverse kinematics using the Jacobian inverse method.
    """
    for iteration in range(N_ITERATIONS):
        current_pos = forward_kinematics(q_now)
        errors, distance = distance_to_goal(current_pos, goal_pos)
        if distance < 0.1:
            print("Solution found in %d iterations." % iteration)
            return q_now, True
        J_inv = jacobian_inverse(q_now)
        q_now = q_now + np.matmul(J_inv, errors)
        print("inv q now: ", q_now)
    return q_now, False


def forward_kinematics(q_now):
        (r1,r2,r3,Uz) = model(q_now, uz_0)
        end_effector = np.array(r1[-1])
        return end_effector


def jacobian_inverse(q_hat):
    # q = np.array([0, 0, 0, 0, np.pi, 0])  # inputs q
    x = np.zeros(3)  # just for size

    r_jac = Jacobian(jac_delta_q, x, q_hat, uz_0, model)
    return r_jac.p_inv()


def distance_to_goal(current_pos, goal_pos):
    x_diff = goal_pos[0] - current_pos[0]
    y_diff = goal_pos[1] - current_pos[1]
    z_diff = goal_pos[2] - current_pos[2]
    return np.array([x_diff, y_diff, z_diff]).T, np.math.sqrt(x_diff**2 + y_diff**2 + z_diff**2)


# def ang_diff(theta1, theta2):
#     """
#     Returns the difference between two angles in the range -pi to +pi
#     """
#     return (theta1 - theta2 + np.pi) % (2 * np.pi) - np.pi


if __name__ == '__main__':
    main()
