'''
    Author: Izzat Kamarudzaman
    Python Version: 3.7.2
    Adapted from Matlab code by Mohsen Khadem

    Code visualises three-tubed concentric tube continuum robot.
'''

import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathos.multiprocessing import ProcessingPool as Pool
from mpl_toolkits.mplot3d import Axes3D
from CTRmodel import moving_CTR
from CTRmodel import plot_3D
from TrajectoryGenerator import TrajectoryGenerator, TrajectoryRetreiver, TrajectoryRetreiverLin


class UzJacobian(object):

    def __init__(self, delta_Uz = [], x = [], q = [], Uz = [], model=0):
        self.delta_Uz = delta_Uz
        self.x = x
        self.q = q
        self.model = model
        self.Uz = Uz
        self.J = np.zeros((len(self.Uz), len(self.Uz)), dtype=np.float)

    def dxdq(self, uz0):
        # return xx.transpose() * qq
        (r1,r2,r3,Uz) = self.model(self.q, uz0)
        # xx = np.array(r1[-1])                               # TODO: check!
        return Uz

    def f(self, uz0):
        return np.array(self.dxdq(uz0), dtype = np.float)

    def parallel_finite_diff(self, i):

        uz0_iplus = self.Uz.copy()
        uz0_iminus = self.Uz.copy()

        uz0_iplus[i] += self.delta_Uz[i] / 2
        uz0_iminus[i] -= self.delta_Uz[i] / 2

        f_plus = self.f(uz0_iplus)
        f_minus = self.f(uz0_iminus)

        return np.array((f_plus - f_minus) / self.delta_Uz[i]).flatten()

    def jac_approx(self):
        for i in np.arange(len(self.Uz)):
            uz_iplus = self.Uz.copy()
            uz_iminus = self.Uz.copy()

            uz_iplus[i] += self.delta_Uz[i] / 2
            uz_iminus[i] -= self.delta_Uz[i] / 2

            f_plus = self.f(uz_iplus)
            f_minus = self.f(uz_iminus)

            self.J[:, i] = np.array((f_plus - f_minus) / self.delta_Uz[i]).flatten()

    def p_inv(self):
        self.jac_approx()
        jac_inv = np.linalg.pinv(self.J)  # inv(X^T * X) * X^T
        return jac_inv


class UzController(object):

    def __init__(self, q, uz0=0, Kp_Uz=5, total_time=1, dt=0.1, 
            model=lambda q,uz:moving_CTR(q,uz), jac_delta_uz=1e-1, 
            plot=False, x_dim = np.zeros(3), parallel=False):
        self.q = q
        self.model = model
        self.Kp_Uz = np.eye(3) * Kp_Uz
        self.total_time = total_time
        self.dt = dt
        self.jac_delta_uz = np.ones(3) * jac_delta_uz
        self.t_steps = int(self.total_time/self.dt)
        self.plot = plot
        self.x_dim = x_dim  # just for size TODO: change to just integer
        self.parallel = parallel

    def get_jinv(self, x_dim, uz0_cur):
        r_jac = UzJacobian(self.jac_delta_uz, self.x_dim, self.q, uz0_cur, self.model)
        return (uz0_cur, r_jac.p_inv())

    # def multiprocess(self, traj, x_des, q_des):
    #     pool = mp.Pool()
    #     results = [pool.apply_async(self.get_jinv, args=(x_des, q_des)) for td in traj]
    #     results = [p.get() for p in results]
    #     results.sort() # to sort the results by input window width
    #     return results

    def run(self):
        t = self.dt
        i = 1
        Uz_traj_pos = np.zeros((3, self.t_steps))
        uz0_start = np.array([0.0, 0.0, 0.0])  # [].transpose()

        uz0_cur_pos = np.zeros((3, self.t_steps))
        Uz_end_cur_pos = np.zeros((3, self.t_steps))

        Uz_traj_vel = np.zeros((3, self.t_steps))
        delta_uz0 = np.zeros((3, self.t_steps))

        delta_Uz = np.zeros((3, self.t_steps))

        (r1,r2,r3,Uz_end) = self.model(self.q, uz0_start)
        Uz_end_cur_pos[:, 0] = Uz_end.flatten()

        a1_coeffs = [[], [], []]
        a2_coeffs = [[], [], []]
        a3_coeffs = [[], [], []]
        waypoints = [Uz_end_cur_pos[:, 0], [0.0, 0.0, 0.0]]

        x = 0  # for x in range(len(waypoints)):
        traj_gen = TrajectoryGenerator(waypoints[x], waypoints[(x + 1) % len(waypoints)], self.total_time)
        traj_gen.solve_lin()  # just 'solve' for quintic
        a1_coeffs[x] = traj_gen.x_c
        a2_coeffs[x] = traj_gen.y_c
        a3_coeffs[x] = traj_gen.z_c

        traj = TrajectoryRetreiverLin()

        Uz_traj_pos[0, 0] = traj.calculate_position(a1_coeffs[0], t)    # FOR INTEGRAL
        Uz_traj_pos[1, 0] = traj.calculate_position(a2_coeffs[0], t)
        Uz_traj_pos[2, 0] = traj.calculate_position(a3_coeffs[0], t)

        while i <= self.t_steps-1:
            # runtime = time.time()
            Uz_traj_pos[0, i] = traj.calculate_position(a1_coeffs[0], t)    # FOR INTEGRAL
            Uz_traj_pos[1, i] = traj.calculate_position(a2_coeffs[0], t)
            Uz_traj_pos[2, i] = traj.calculate_position(a3_coeffs[0], t)

            Uz_traj_vel[0, i] = traj.calculate_velocity(a1_coeffs[0], t)   # 0.9 for final velocity
            Uz_traj_vel[1, i] = traj.calculate_velocity(a2_coeffs[0], t)
            Uz_traj_vel[2, i] = traj.calculate_velocity(a3_coeffs[0], t)

            delta_Uz[:, i] = Uz_traj_pos[:, i] - Uz_end_cur_pos[:, i-1]
            # delta_Uz[:, i] = np.array([0,0,0]) - Uz_end_cur_pos[:, i-1]

            # get trajectory from Jacobian
            r_jac = UzJacobian(self.jac_delta_uz, self.x_dim, self.q, uz0_cur_pos[:, i-1], self.model)

            if self.parallel:
                pickled_finite_diff = lambda i:r_jac.parallel_finite_diff(i)
                pool = Pool()

                results = pool.map(pickled_finite_diff, range(len(uz0_start)))
                parallel_J = np.zeros([len(uz0_start), len(uz0_start)], dtype=np.float)
                for x in np.arange(len(uz0_start)):
                    parallel_J[:, x] = results[x]
                J_inv = parallel_J.transpose() @ np.linalg.inv(parallel_J@parallel_J.transpose())
            else:
                J_inv = r_jac.p_inv()

            delta_uz0[:, i] = J_inv @ (Uz_traj_vel[:, i] + self.Kp_Uz @ delta_Uz[:, i])

            uz0_cur_pos[:, i] = uz0_cur_pos[:, i-1] + delta_uz0[:, i].copy() * self.dt
            # Uz_end_cur_pos += Uz_traj_vel[:, i].copy() * dt  # TODO: change to Forward Kinematics
            (r1,r2,r3,Uz) = self.model(self.q, uz0_cur_pos[:, i])
            Uz_end_cur_pos[:, i] = Uz.flatten()

            # print(i, time.time()-runtime)
            t += self.dt
            i += 1

        # print('Uz_end_cur_pos', Uz_end_cur_pos[:, -1])
        if self.plot:
            plt.subplots(1)
            tt = np.arange(0.0, self.total_time, self.dt)
            plt.plot(tt, Uz_end_cur_pos[0], label='1')
            plt.plot(tt, Uz_end_cur_pos[1], label='2')
            plt.plot(tt, Uz_end_cur_pos[2], label='3')
            plt.title('Uz_end_cur_pos')
            plt.legend()

            # plt.subplots(1)
            # # tt = np.arange(0.0, self.total_time, self.dt)
            # plt.plot(tt, delta_Uz[0], label='1')
            # plt.plot(tt, delta_Uz[1], label='2')
            # plt.plot(tt, delta_Uz[2], label='3')
            # plt.title('delta_Uz')
            # plt.legend()

            # plt.subplots(1)
            # # tt = np.arange(0.0, self.total_time, self.dt)
            # plt.plot(tt, Uz_traj_pos[0], label='1')
            # plt.plot(tt, Uz_traj_pos[1], label='2')
            # plt.plot(tt, Uz_traj_pos[2], label='3')
            # plt.title('Uz_traj_pos')

            # plt.subplots(1)
            # plt.plot(tt, Uz_traj_vel[0])
            # plt.plot(tt, Uz_traj_vel[1])
            # plt.plot(tt, Uz_traj_vel[2])
            # plt.title('uz0 vel trajectory')

            # plt.subplots(1)
            # plt.plot(tt, uz0_cur_pos[0])
            # plt.plot(tt, uz0_cur_pos[1])
            # plt.plot(tt, uz0_cur_pos[2])
            # plt.title('uz0_cur_pos trajectory')
        # print(t)
        return uz0_cur_pos[:, -1]

    def Uz_controlled_model(self):
        # runtime = time.time()
        controlled_uz0 = self.run() 
        # xxx = self.model(self.q, controlled_uz0)
        # print("UzRunTime:", time.time()-runtime)
        # print("new Uz0:", controlled_uz0)
        return self.model(self.q, controlled_uz0)

    # def calculate_position(self, c, t):
    #     return t + c[0]

    # def calculate_velocity(self, c, t):
    #     return 0.2

if __name__ == "__main__":

    a_ans = (2*np.pi)/3
    q_static = np.array([0.001, 0.001, 0.001, 0.001, a_ans, a_ans])
    uz0_start = np.array([[0.0, 0.0, 0.0]]).transpose()
    ax = plt.axes(projection='3d')

    runtime = time.time()
    (r1,r2,r3,Uz) = moving_CTR(q_static, uz0_start)
    print("\nNormalRuntime:", time.time()-runtime)

    plot_3D(ax, r1, r2, r3, 'init position')
    print('Uz before: ', Uz)

    # uz0_cont = UzController(q_static)
    # uz0_new = uz0_cont.run()

    runtime = time.time()
    (r1,r2,r3,Uz) = UzController(q_static, uz0_start, plot=True, dt=0.2, parallel=True, 
                    jac_delta_uz=0.1, Kp_Uz=5).Uz_controlled_model()  # moving_CTR(q_static, uz0_new)
    print("\nUzRunTime:", time.time()-runtime)

    plot_3D(ax, r1, r2, r3, 'final position')
    print('Uz after: ', Uz)

    # # Create cubic bounding box to simulate equal aspect ratio
    # max_range = 0.2  # np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
    # Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(0)  # X.max()+X.min())
    # Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(0)  # Y.max()+Y.min())
    # Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(0.3)  # Z.max()+Z.min())
    # # Comment or uncomment following both lines to test the fake bounding box:
    # for xb, yb, zb in zip(Xb, Yb, Zb):
    #     ax.plot([xb], [yb], [zb], 'w')

    ax.legend()
    ax.set_xlabel('tube1')
    ax.set_ylabel('tube2')
    ax.set_zlabel('tube3')

    plt.show()


    # import CTRmodel
    # model = lambda q, uz_0: CTRmodel.moving_CTR(q, uz_0)

    # uz_0 = np.array([[0.0, 0.0, 0.0]]).transpose()
    # uz0_start = np.array([0.0, 0.0, 0.0])
    # q = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # inputs q
    # delta_Uz = np.ones(3) * 1e-3
    # x = np.zeros(3)

    # u_jac = UzJacobian(delta_Uz, x, q, uz_0, model)
    # # u_jac.jac_approx()
    # J = u_jac.J
    # J_inv = u_jac.p_inv()

    # print('J:\n', J)
    # print('\nJ_inv:\n', J_inv)
    # print('\na * a+ * a == a   -> ', np.allclose(J, np.dot(J, np.dot(J_inv, J))))
    # print('\na+ * a * a+ == a+ -> ', np.allclose(J_inv, np.dot(J_inv, np.dot(J, J_inv))))

    # r_jac_p = UzJacobian(delta_Uz, x, q, uz_0, model)
    # pickled_finite_diff = lambda i:r_jac_p.parallel_finite_diff(i)
    # pool = Pool()

    # results = pool.map(pickled_finite_diff, range(len(uz0_start)))
    # parallel_J = np.zeros([len(uz0_start), len(uz0_start)], dtype=np.float)
    # for i in np.arange(len(uz0_start)):
    #     parallel_J[:, i] = results[i]
    # J_inv_p = parallel_J.transpose() @ np.linalg.inv(parallel_J@parallel_J.transpose())

    # print('J:\n', parallel_J)
    # print('\nJ_inv:\n', J_inv_p)
    # print('\na * a+ * a == a   -> ', np.allclose(parallel_J, np.dot(parallel_J, np.dot(J_inv_p, parallel_J))))
    # print('\na+ * a * a+ == a+ -> ', np.allclose(J_inv_p, np.dot(J_inv_p, np.dot(parallel_J, J_inv_p))))