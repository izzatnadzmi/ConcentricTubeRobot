'''
    Author: Izzat Kamarudzaman
    Python Version: 3.7.2
    Adapted from Matlab code by Mohsen Khadem

    Code visualises three-tubed concentric tube continuum robot.
'''

import numpy as np
import sys
import time
sys.path.append("../")
sys.path.append("./ConcentricTubeRobot/")
sys.path.append("./ConcentricTubeRobot/test")
from pathos.multiprocessing import ProcessingPool as Pool
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from CTRmodel import moving_CTR
from CTR_model import CTRobotModel, plot_3D
from TrajectoryGenerator import TrajectoryGenerator, TrajectoryRetreiver, TrajectoryRetreiverLin
from trajTest import HelicalGenerator
from functools import partial


# ##########################################################################################
class UzJacobian(object):

    def __init__(self, delta_Uz = [], x = [], q = [], Uz = [], model=0):
        self.delta_Uz = delta_Uz
        self.x = x
        self.q = q
        self.model = model
        self.Uz = Uz
        self.J = np.zeros((len(self.Uz), len(self.Uz)), dtype=np.float)

    def f(self, uz0):
        (r1,r2,r3,Uz) = self.model(self.q, uz0)
        return np.array(Uz, dtype = np.float)

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

        while i <= self.t_steps-1:
            # runtime = time.time()
            Uz_traj_pos[0, i] = traj.calculate_position(a1_coeffs[0], 0.9)    # FOR INTEGRAL
            Uz_traj_pos[1, i] = traj.calculate_position(a2_coeffs[0], 0.9)
            Uz_traj_pos[2, i] = traj.calculate_position(a3_coeffs[0], 0.9)

            Uz_traj_vel[0, i] = traj.calculate_velocity(a1_coeffs[0], 0.9)
            Uz_traj_vel[1, i] = traj.calculate_velocity(a2_coeffs[0], 0.9)
            Uz_traj_vel[2, i] = traj.calculate_velocity(a3_coeffs[0], 0.9)

            # delta_Uz[:, i] = Uz_traj_pos[:, i] - Uz_end_cur_pos[:, i-1]
            delta_Uz[:, i] = np.array([0,0,0]) - Uz_end_cur_pos[:, i-1]

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

            plt.subplots(1)
            tt = np.arange(0.0, self.total_time, self.dt)
            plt.plot(tt, delta_Uz[0], label='1')
            plt.plot(tt, delta_Uz[1], label='2')
            plt.plot(tt, delta_Uz[2], label='3')
            plt.title('delta_Uz')
            plt.legend()

            # # plt.subplots(1)
            # # tt = np.arange(0.0, self.total_time, self.dt)
            # # plt.plot(tt, traj.calculate_position(a1_coeffs[0], tt))
            # # plt.plot(tt, traj.calculate_position(a2_coeffs[0], tt))
            # # plt.plot(tt, traj.calculate_position(a3_coeffs[0], tt))
            # # plt.title('uz0 pos trajectory')

            plt.subplots(1)
            plt.plot(tt, Uz_traj_vel[0])
            plt.plot(tt, Uz_traj_vel[0])
            plt.plot(tt, Uz_traj_vel[0])
            plt.title('uz0 vel trajectory')

        return uz0_cur_pos[:, -1]

    def Uz_controlled_model(self):
        # runtime = time.time()
        controlled_uz0 = self.run() 
        # xxx = self.model(self.q, controlled_uz0)
        # print("UzRunTime:", time.time()-runtime)
        return self.model(self.q, controlled_uz0)

# ##########################################################################################

class CombinedJacobian(object):
    def __init__(self, delta_q = [], x = [], q = [], uz_0 = [], model=0, damped_lsq=0):
        self.delta_q = delta_q
        self.x = x
        self.q = q
        self.model = model
        self.uz_0 = uz_0
        self.J = np.zeros([len(self.x), len(self.q)], dtype=np.float)
        self.damped_lsq = damped_lsq

    def f(self, qq):
        (r1,r2,r3,Uz) = self.model(qq, self.uz_0)
        xx = np.array(r1[-1])
        return np.array([xx], dtype = np.float)

    def parallel_finite_diff(self, i):
        if i < 6:
            q_iplus = self.q.copy()
            q_iminus = self.q.copy()
            q_iplus[i] += self.delta_q[i] / 2
            q_iminus[i] -= self.delta_q[i] / 2

            f_plus = self.f(q_iplus)
            f_minus = self.f(q_iminus)
        if i >= 6 & i < 9:


        return (f_plus - f_minus) / self.delta_q[i]

    def jac_approx(self):
        for i in np.arange(len(self.q)):
            q_iplus = self.q.copy()
            q_iminus = self.q.copy()

            q_iplus[i] += self.delta_q[i] / 2
            q_iminus[i] -= self.delta_q[i] / 2

            f_plus = self.f(q_iplus)
            f_minus = self.f(q_iminus)

            self.J[:, i] = (f_plus - f_minus) / self.delta_q[i]

    def p_inv(self):
        self.jac_approx()
        jac_inv = self.J.transpose()@np.linalg.inv(self.J@self.J.transpose() + self.damped_lsq*np.eye(3))
        #     jac_inv = np.linalg.pinv(self.J)  # inv(X^T * X) * X^T
        return jac_inv


class Parallelisor(object):
    def __init__(self, jac_del_q, x, q, uz_0, model):
        self.jac_del_q = jac_del_q
        self.x = x
        self.q = q
        self.uz_0 = uz_0
        self.model = model

    def combined_jac(self):                       #TODO #TODO #TODO #TODO #TODO #TODO #TODO #TODO #TODO #TODO #TODO #TODO #TODO #TODO #TODO #TODO #TODO #TODO
        r_jac = CombinedJacobian(self.jac_del_q, self.x, self.q, self.uz_0, self.model)
        
        pickled_finite_diff = lambda i:r_jac.parallel_finite_diff(i)
        pool = Pool()

        results = pool.map(pickled_finite_diff, range(len(self.q)))
        parallel_J = np.zeros([len(self.x), len(self.q)], dtype=np.float)
        for x in np.arange(len(self.q)):
            parallel_J[:, x] = results[x]
        # J_inv = parallel_J.transpose() @ np.linalg.inv(parallel_J@parallel_J.transpose())
        return parallel_J


class CombinedController(object):
    def __init__(self, Kp_x=5, Ki_x=0, Kd_x=0, total_time=1, dt=0.05, sim=False,
            model=lambda q,uz:moving_CTR(q,uz), plot=False, vanilla_model=None,
            jac_del_q=1e-1, damped_lsq=0, pertubed_model=None, parallel=False, helical=None):
        self.model = model
        self.vanilla_model = vanilla_model
        self.Kp_x = np.eye(3) * Kp_x
        self.Ki_x = np.eye(3) * Ki_x
        self.Kd_x = np.eye(3) * Kd_x
        self.total_time = total_time
        self.dt = dt
        self.jac_del_q = np.ones(6) * jac_del_q
        self.t_steps = int(self.total_time/self.dt)
        self.result = {}
        self.plot = plot
        self.sim = sim
        self.damped_lsq = damped_lsq
        self.parallel = parallel
        self.helical = helical
        if pertubed_model:
            self.pertubed_model = pertubed_model
            print('Using Pertubed Model!')
        else:
            self.pertubed_model = model
            print('NOT Using Pertubed Model!')

    def get_jinv(self, x_des, q_des):
        r_jac = CombinedJacobian(self.jac_del_q, x_des, q_des, uz_0, self.model, self.damped_lsq)
        return (q_des, r_jac.p_inv())

    # def multiprocess(self, traj, x_des, q_des):
        # pool = mp.Pool()
        # results = [pool.apply_async(self.get_jinv, args=(x_des, q_des)) for td in traj]
        # results = [p.get() for p in results]
        # results.sort() # to sort the results by input window width
        # return results

    def run(self, a1_c, a2_c, a3_c, q_start, x_end_pos):
        runtime = time.time()
        t = self.dt
        i = 1

        q_des_pos = np.zeros((6, self.t_steps))  # [BBBaaa]
        x_des_pos = np.zeros((3, self.t_steps))  # [r]
        x_des_vel = np.zeros((3, self.t_steps))  # [r]
        x_cur_pos = np.zeros((3, self.t_steps))  # [r]

        delta_x = np.zeros((3, self.t_steps))  # [r]
        delta_v = np.zeros((3, self.t_steps))  # [r]
        delta_q = np.zeros((6, self.t_steps))  # [BBBaaa]
        # delta_x = np.zeros(3)  # [r]
        integral = 0.0

        if self.helical:
            quintic = self.helical
            # gettraj_pos = lambda aa, tt : quintic.calculate_position(aa, tt)
            # gettraj_vel = lambda aa, tt : quintic.calculate_velocity(aa, tt)

            # for aa in range(3):
            #     func_pos = partial(gettraj_pos, aa)
            #     pooltraj = Pool()
            #     traj_results = pooltraj.map(func_pos, np.arange(0.0, self.total_time, self.dt))
            #     x_des_pos[aa, :] = traj_results

            # for aa in range(3):
            #     func_pos = partial(gettraj_vel, aa)
            #     pooltraj = Pool()
            #     traj_results = pooltraj.map(func_pos, np.arange(0.0, self.total_time, self.dt))
            #     x_des_vel[aa, :] = traj_results

        else:
            quintic = TrajectoryRetreiverLin()
            # quintic = TrajectoryRetreiver()

        q_des_pos[:, 0] = q_start
        x_des_pos[0, 0] = quintic.calculate_position(a1_c[0], t)
        x_des_pos[1, 0] = quintic.calculate_position(a2_c[0], t)
        x_des_pos[2, 0] = quintic.calculate_position(a3_c[0], t)
        x_cur_pos[:, 0] = x_des_pos[:, 0]

        if self.sim:
            x_sim_pos = np.zeros((3, self.t_steps))  # [r]
            x_sim_pos[:, 0] = x_des_pos[:, 0]

        ax = plt.axes(projection='3d')

        while i < self.t_steps:
            x = np.zeros(3)  # just for size TODO: change to just integer
            # if not self.helical:
            x_des_pos[0, i] = quintic.calculate_position(a1_c[0], t)
            x_des_pos[1, i] = quintic.calculate_position(a2_c[0], t)
            x_des_pos[2, i] = quintic.calculate_position(a3_c[0], t)
            x_des_vel[0, i] = quintic.calculate_velocity(a1_c[0], t)
            x_des_vel[1, i] = quintic.calculate_velocity(a2_c[0], t)
            x_des_vel[2, i] = quintic.calculate_velocity(a3_c[0], t)

            delta_x[:, i] = x_des_pos[:, i] - x_cur_pos[:, i-1]
            # delta_x[:, i] = x_end_pos - x_cur_pos[:, i-1]
            # delta_v[:, i] = x_des_vel[:, i] - x_des_vel[:, i-1]
            # delta_v[:, i] = (delta_x[:, i] - delta_x[:, i-1])/dt
            integral += delta_x[:, i] * self.dt

            # get trajectory from Jacobian
            r_jac = CombinedJacobian(self.jac_del_q, x, q_des_pos[:, i-1].flatten(), uz_0, self.model, self.damped_lsq)

            if self.parallel:
                pickled_finite_diff = lambda i:r_jac.parallel_finite_diff(i)
                pool = Pool()

                results = pool.map(pickled_finite_diff, range(len(q_start)))
                parallel_J = np.zeros([len(x), len(q_start)], dtype=np.float)
                for x in np.arange(len(q_start)):
                    parallel_J[:, x] = results[x]
                J_inv = parallel_J.transpose() @ np.linalg.inv(parallel_J@parallel_J.transpose())

            else:
                J_inv = r_jac.p_inv()

            delta_q[:, i] = J_inv @ ((x_des_vel[:, i]) + self.Kp_x@(delta_x[:, i]) +  # (x_des_vel[:, i]-x_des_vel[:, i-1])
                                        self.Ki_x@integral + self.Kd_x@(delta_v[:, i]))
            q_des_pos[:, i] = q_des_pos[:, i-1] + delta_q[:, i] * self.dt
            (r1,r2,r3,Uz) = self.pertubed_model(q_des_pos[:, i], uz_0)        # FORWARD KINEMATICS
            if i%2000 == 0:
                plot_3D(ax, r1, r2, r3)
            x_cur_pos[:, i] = r1[-1]

            if self.sim:
                (r1,r2,r3,Uz) = self.vanilla_model(q_des_pos[:, i], uz_0)
                x_sim_pos[:, i] = r1[-1]

            print('TimeStep:', t)#, 'RunTime:', time.time()-runtime)
            # print('i:', i, 'tsteps:', self.t_steps)
            t += self.dt
            i += 1

        
        print('------------------------------------------')
        print('Kp_x:', self.Kp_x)
        mx = np.argmax(delta_x[0])
        my = np.argmax(delta_x[1])
        mz = np.argmax(delta_x[2])
        print('Max x', delta_x[:, mx])
        print('Max y', delta_x[:, my])
        print('Max z', delta_x[:, mz])
        print('Final Error', delta_x[:, -1])
        print('Mean X Error', delta_x[0])
        print('Mean Y Error', delta_x[1])
        print('Mean Z Error', delta_x[2])
        print("Full Run:", time.time()-runtime)
        self.result[str(self.Kp_x)] = max([mx, my, mz])
        print('------------------------------------------')

        if self.plot:
            # fig = plt.figure()
            # ax = plt.axes(projection='3d')
            # colors = cm.rainbow(np.linspace(0, 1, len(x_cur_pos.transpose())))
            # for y, c in zip(x_cur_pos.transpose(), colors):
            #     # plt.scatter(x, y, color=c)
            #     ax.scatter(y[0], y[1], y[2], linewidth=1, color=c)
            ax.plot3D(x_cur_pos[0], x_cur_pos[1], x_cur_pos[2], linewidth=2, label='Actual Trajectory', color='red')
            ax.plot3D(x_des_pos[0], x_des_pos[1], x_des_pos[2], linewidth=1, label='Desired Trajectory', color='green', linestyle='--')
            if self.sim:
                ax.plot3D(x_sim_pos[0], x_sim_pos[1], x_sim_pos[2], linewidth=1, label='Traj. w/o Uz Controller', linestyle=':', color='black')

            # (r1,r2,r3,Uz) = self.model(q_des_pos[:, 0], uz_0)
            # plot_3D(ax, r1, r2, r3, 'initial pos')
            # (r1,r2,r3,Uz) = self.model(q_des_pos[:, -1], uz_0)
            # plot_3D(ax, r1, r2, r3, 'final pos - q w/ controller')
            ax.legend()
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            # plt.axis('equal')
            # ax.set_aspect('equal')

            # Create cubic bounding box to simulate equal aspect ratio
            max_range = 0.1  # np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
            Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(0)  # X.max()+X.min())
            Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(0)  # Y.max()+Y.min())
            Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(0.3)  # Z.max()+Z.min())
            # Comment or uncomment following both lines to test the fake bounding box:
            for xb, yb, zb in zip(Xb, Yb, Zb):
                ax.plot([xb], [yb], [zb], 'w')
            # plt.savefig('traj.png')

            plt.subplots(1)
            tt = np.arange(0.0, self.total_time, self.dt)
            plt.plot(tt, q_des_pos[0], label='q_B1')
            plt.plot(tt, q_des_pos[1], label='q_B2')
            plt.plot(tt, q_des_pos[2], label='q_B3')
            plt.plot(tt, q_des_pos[3], label='q_a1')
            plt.plot(tt, q_des_pos[4], label='q_a2')
            plt.plot(tt, q_des_pos[5], label='q_a3')
            plt.title('q inputs')
            plt.legend()
            # plt.savefig('qin.png')

            plt.subplots(1)
            tt = np.arange(0.0, self.total_time, self.dt)
            plt.plot(tt, delta_x[0], label='x')
            plt.plot(tt, delta_x[1], label='y')
            plt.plot(tt, delta_x[2], label='z')
            plt.title('delta_x')
            plt.legend()
            # plt.savefig('delx.png')

            plt.subplots(1)
            tt = np.arange(0.0, self.total_time, self.dt)
            plt.plot(tt, delta_q[0], label='del_q.B1')
            plt.plot(tt, delta_q[1], label='del_q.B2')
            plt.plot(tt, delta_q[2], label='del_q.B3')
            plt.plot(tt, delta_q[3], label='del_q.a1')
            plt.plot(tt, delta_q[4], label='del_q.a2')
            plt.plot(tt, delta_q[5], label='del_q.a3')
            plt.title('delta_q')
            plt.legend()
            # plt.savefig('delq.png')
            
            plt.subplots(1)
            plt.plot(tt, x_des_vel[0, :], label='x')
            plt.plot(tt, x_des_vel[1, :], label='y')
            plt.plot(tt, x_des_vel[2, :], label='z')
            plt.title('x_des_vel trajectory')
            plt.legend()
            # plt.savefig('xvel.png')

            plt.subplots(1)
            plt.plot(tt, x_des_pos[0, :], label='X trajectory', linestyle=':')
            plt.plot(tt, x_des_pos[1, :], label='Y trajectory', linestyle=':')
            plt.plot(tt, x_des_pos[2, :], label='Z trajectory', linestyle=':')
            plt.plot(tt, x_cur_pos[0, :], label='X traj. w/ Int. Controller', linestyle='--')
            plt.plot(tt, x_cur_pos[1, :], label='Y traj. w/ Int. Controller', linestyle='--')
            plt.plot(tt, x_cur_pos[2, :], label='Z traj. w/ Int. Controller', linestyle='--')
            if self.sim:
                plt.plot(tt, x_sim_pos[0, :], label='sim.x')
                plt.plot(tt, x_sim_pos[1, :], label='sim.y')
                plt.plot(tt, x_sim_pos[2, :], label='sim.z')
            plt.title('x_des_pos/x_cur_pos trajectory')
            plt.legend()
            # plt.savefig('xtraj.png')

            plt.show()


def uncertainty(value, error=0.05):
    noise = np.random.choice([-1, 1], (value.shape)) * value * error
    return value + noise



if __name__ == "__main__":

# MAIN
    # a_ans = (2*np.pi)/4
    # total_time = 1
    # dt = 0.01
    # Uzdt = 0.1
    # UzControl = True
    # jac_del_q = 1e-3
    # Kp_x = 10
    # damped_lsq = 0.0
    # perturbed = False
    # parallel = True
    # Uz_parallel = False
    # helical = True
    # sim = True
    # print('Damped least square for Jacobian:', damped_lsq)

    # no_of_tubes = 3  # ONLY WORKS FOR 3 TUBES for now
    # initial_q = [-0.2858, -0.2025, -0.0945, 0, 0, 0]
    # tubes_length = 1e-3 * np.array([431, 332, 174])              # length of tubes
    # curve_length = 1e-3 * np.array([103, 113, 134])              # length of the curved part of tubes


    # # physical parameters
    # E = np.array([ 6.4359738368e+10, 5.2548578304e+10, 4.7163091968e+10])   # E stiffness
    # J = 1.0e-11 * np.array([0.0120, 0.0653, 0.1686])                        # J second moment of inertia
    # I = 1.0e-12 * np.array([0.0601, 0.3267, 0.8432])                        # I inertia
    # G = np.array([2.5091302912e+10, 2.1467424256e+10, 2.9788923392e+10] )   # G torsion constant

    # Ux = np.array([21.3, 13.108, 3.5])                  # constant U curvature vectors for each tubes
    # Uy = np.array([0, 0, 0])

    # ctr = CTRobotModel(no_of_tubes, tubes_length, curve_length, initial_q, E, J, I, G, Ux, Uy)


    # # physical parameters  # PERTURBED
    # # E = np.array([ 6.4359738368e+10, 5.2548578304e+10, 4.7163091968e+10])  # E stiffness
    # # E = uncertainty(E)
    # E = np.array([6.11417514e+10, 4.99211494e+10, 4.48049374e+10])
    # J = 1.0e-11 * np.array([0.0120, 0.0653, 0.1686])                         # J second moment of inertia
    # I = 1.0e-12 * np.array([0.0601, 0.3267, 0.8432])                         # I inertia
    # # G = np.array([2.5091302912e+10, 2.1467424256e+10, 2.9788923392e+10] )  # G torsion constant
    # # G = uncertainty(G)
    # G = np.array([2.63458681e+10, 2.25407955e+10, 2.82994772e+10])

    # Ux = np.array([21.3, 13.108, 3.5])                  # constant U curvature vectors for each tubes
    # Uy = np.array([0, 0, 0])

    # ctr_perturbed = CTRobotModel(no_of_tubes, tubes_length, curve_length, initial_q, E, J, I, G, Ux, Uy)


    # # (r1,r2,r3,Uz) = ctr.moving_CTR(q, uz_0)
    # ctr_model = lambda q,uz:ctr.moving_CTR(q,uz)
    # ctr_perturbed_model = lambda q,uz:ctr_perturbed.moving_CTR(q,uz)

    # if UzControl:
    #     model = lambda q,uz:UzController(q,uz, dt=Uzdt, model=ctr_model, parallel=Uz_parallel).Uz_controlled_model()
    #     print('Using Uz Controller!')
    # else:
    #     model = ctr_model
    #     print('NOT Using Uz Controller!')

    # if helical:
    #     q_start = np.array([0.0101, 0.0101, 0.0101, -a_ans, -a_ans, -a_ans])  # a_ans, a_ans, a_ans
    #     q_end = np.array([0.0101, 0.0101, 0.0101, a_ans, a_ans, a_ans])  # ([1.0001, -1.0001, 0.7001, a_ans + 0.2, a_ans + 0.2, a_ans + 0.2])
    # else:
    #     q_start = np.array([-0.0101, -0.0101, -0.0101, a_ans/2, a_ans/2, a_ans/2])  # a_ans, a_ans, a_ans
    #     q_end = np.array([0.0101, 0.0101, 0.0101, a_ans, a_ans, a_ans])  # ([1.0001, -1.0001, 0.7001, a_ans + 0.2, a_ans + 0.2, a_ans + 0.2])

    # uz_0 = np.array([0.0, 0.0, 0.0])
    # (r1,r2,r3,Uz) = model(q_start, uz_0)
    # x_cur_pos = r1[-1]
    # (r1e,r2e,r3e,Uze) = model(q_end, uz_0)
    # x_end_pos = r1e[-1]

    # waypoints = [x_cur_pos, x_end_pos]
    # a1_coeffs = []
    # a2_coeffs = []
    # a3_coeffs = []

    # if helical:
    #     a1_coeffs = [0]*int(total_time/dt)
    #     a2_coeffs = [1]*int(total_time/dt)
    #     a3_coeffs = [2]*int(total_time/dt)
    #     helical_traj = HelicalGenerator(x_cur_pos, [-0.25, -0.25, 0.25])
    # else:
    #     helical_traj = None
    #     for x in range(len(waypoints)):
    #         traj = TrajectoryGenerator(waypoints[x], waypoints[(x + 1) % len(waypoints)], total_time)
    #         traj.solve_lin()
    #         # traj.solve()
    #         a1_coeffs.append(traj.x_c)
    #         a2_coeffs.append(traj.y_c)
    #         a3_coeffs.append(traj.z_c)

    # if perturbed:
    #     CTR_sim = CombinedController(Kp_x=Kp_x, model=model, total_time=total_time, dt=dt, jac_del_q=jac_del_q, 
    #                             plot=True, damped_lsq=damped_lsq, pertubed_model = ctr_perturbed_model,
    #                             parallel=parallel, vanilla_model=ctr_model, sim=sim, helical=helical_traj)
    # else:
    #     CTR_sim = CombinedController(Kp_x=Kp_x, model=model, total_time=total_time, dt=dt, jac_del_q=jac_del_q, 
    #                             plot=True, damped_lsq=damped_lsq, parallel=parallel, 
    #                             vanilla_model=ctr_model, sim=sim, helical=helical_traj)
    # CTR_sim.run(a1_coeffs, a2_coeffs, a3_coeffs, q_start, x_end_pos)

    # result = CTR_sim.result
    # print(max(result, key=lambda key: result[key]))


# JACOBIAN TEST
    import CTRmodel
    model = lambda q, uz_0: CTRmodel.moving_CTR(q, uz_0)

    uz_0 = np.array([0.0, 0.0, 0.0])
    q = np.array([0.0, 0.0, 0.0, 0.0, np.pi, np.pi])  # inputs q
    delta_q = np.ones(6) * 1e-3
    x = np.zeros(3)

    runtime = time.time()
    u_jac = CombinedJacobian(delta_q, x, q, uz_0, model)
    u_jac.jac_approx()
    J = u_jac.J
    J_inv = u_jac.p_inv()

    print('RunTime:', time.time()-runtime)
    print('J:\n', J)
    print('\nJ_inv:\n', J_inv)
    print('\na * a+ * a == a   -> ', np.allclose(J, np.dot(J, np.dot(J_inv, J))))
    print('\na+ * a * a+ == a+ -> ', np.allclose(J_inv, np.dot(J_inv, np.dot(J, J_inv))))

    # damping_factor = 0.5
    # # damped_pinv = np.linalg.inv(J.transpose()@J + damping_factor*np.eye(6))@J.transpose()
    # print(J)
    # damped_pinv = J.transpose()@np.linalg.inv(J@J.transpose() + damping_factor*np.eye(3))
    # print('\ndamped_pinv:\n', damped_pinv)
    # print('\na * a+ * a == a   -> ', np.allclose(J, np.dot(J, np.dot(damped_pinv, J))))
    # print('\na+ * a * a+ == a+ -> ', np.allclose(damped_pinv, np.dot(damped_pinv, np.dot(J, damped_pinv))))

    runtime = time.time()
    parallel_jac = Parallelisor(delta_q, x, q, uz_0, model)
    # pickled_finite_diff = lambda i:parallel_jac.parallel_finite_diff(i)

    # pool = Pool()
    # results = pool.map(pickled_finite_diff, range(len(q)))

    # parallel_J = np.zeros([len(x), len(q)], dtype=np.float)
    # for i in np.arange(len(q)):
    #     parallel_J[:, i] = results[i]
    
    parallel_J = parallel_jac.combined_jac()

    parallel_jac_inv = parallel_J.transpose() @ np.linalg.inv(parallel_J@parallel_J.transpose())

    print('RunTime:', time.time()-runtime)
    print('parallel_J:\n', parallel_J)
    print('\nparallel_jac_inv:\n', parallel_jac_inv)
    print('\na * a+ * a == a   -> ', np.allclose(parallel_J, np.dot(parallel_J, np.dot(parallel_jac_inv, parallel_J))))
    print('\na+ * a * a+ == a+ -> ', np.allclose(parallel_jac_inv, np.dot(parallel_jac_inv, np.dot(parallel_J, parallel_jac_inv))))
