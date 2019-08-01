import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class HelicalGenerator():
    def __init__(self, start_pos, des_pos, # total_time, dt, z_max=0.01, 
                    start_vel=[0,0,0], des_vel=[0,0,0]):
        # self.theta = 0
        self.x = 0
        self.y = 0
        self.z = 0
        # self.dt = dt
        # self.z_max = z_max
        # self.r = self.theta
        # self.total_time = total_time

        self.x1 = start_pos[0]
        self.y1 = start_pos[1]
        self.z1 = start_pos[2]

        self.x2 = des_pos[0]
        self.y2 = des_pos[1]
        self.z2 = des_pos[2]

        self.start_x_vel = start_vel[0]
        self.start_y_vel = start_vel[1]
        self.start_z_vel = start_vel[2]

        self.des_x_vel = des_vel[0]
        self.des_y_vel = des_vel[1]
        self.des_z_vel = des_vel[2]

        self.d = np.sqrt((self.x1 - self.x2)**2 + (self.y1 - self.y2)**2)
        self.t0 = np.tan((self.y2 - self.y1)/(self.x1 - self.x2))
        self.rev = 1
        self.m = 1

    def helical_traj(self, t):
        # self.theta = t
        # self.r = self.theta/30
        # self.x = 1.25 * self.r*np.cos(self.theta)  
        # self.y = 1.25 * self.r*np.sin(self.theta)
        # self.z = 0.2 + self.z_max*self.theta
        self.x = self.x1 + self.m * t * self.d * np.cos(2 * np.pi * self.rev * t + self.t0)
        self.y = self.y1 + self.m * t * self.d * np.sin(2 * np.pi * self.rev * t + self.t0)
        self.z = self.z1 + t * (self.z2 - self.z1)

    def calculate_position(self, c, t):
        self.helical_traj(t)
        if c == 0:
            return self.x
        if c == 1:
            return self.y
        if c == 2:
            return self.z

    def calculate_velocity(self, c, t):
        if c == 0:
            return (2 * t*self.d * -np.sin(2*np.pi*self.rev*t+self.t0)*(2*np.pi*self.rev)) + \
                        (np.cos(2*np.pi*self.rev*t+self.t0) * self.m * self.d)
        if c == 1:
            return (2 * t*self.d * np.cos(2*np.pi*self.rev*t+self.t0)*(2*np.pi*self.rev)) + \
                        (np.sin(2*np.pi*self.rev*t+self.t0) * self.m * self.d)
        if c == 2:
            return self.z2 - self.z1

    # def helical_getVel(self):
    #     self.x = self.r*np.cos(self.theta)  
    #     self.y = self.r*np.sin(self.theta)
    #     self.z = self.z_max*self.theta

    # def helical_getTraj(self, t):
    #     theta = np.radians(np.linspace(180, 180*2.5, int(self.total_time/self.dt)))
    #     return self.helical_traj(theta[t])


if __name__ == "__main__":
    x_2 = []
    y_2 = []
    z_2 = []

    x_v = []
    y_v = []
    z_v = []
    # hell = TrajectoryGenerator()
    # ax = plt.axes(projection='3d')
    # theta = np.radians(np.linspace(180,180*2.5,1000))
    # for xx in theta:
    #     hell.helical_traj(xx)
    #     x_2.append(hell.x)
    #     y_2.append(hell.y)
    #     z_2.append(hell.z)

    # hell = HelicalGenerator(1, 0.001)
    # ax = plt.axes(projection='3d')
    # for xx in np.arange(1000):
    #     hell.helical_getTraj(xx)
    #     x_2.append(hell.x)
    #     y_2.append(hell.y)
    #     z_2.append(hell.z)

    import os
    print(os.getcwd())
    import sys
    sys.path.append("../")
    sys.path.append("./ConcentricTubeRobot/")
    from CurvatureController import UzController
    from CTR_model import CTRobotModel, plot_3D

    no_of_tubes = 3  # ONLY WORKS FOR 3 TUBES for now
    initial_q = [-0.2858, -0.2025, -0.0945, 0, 0, 0]
    tubes_length = 1e-3 * np.array([431, 332, 174])              # length of tubes
    curve_length = 1e-3 * np.array([103, 113, 134])              # length of the curved part of tubes
    Uzdt = 0.1

    # physical parameters
    E = np.array([ 6.4359738368e+10, 5.2548578304e+10, 4.7163091968e+10])   # E stiffness
    J = 1.0e-11 * np.array([0.0120, 0.0653, 0.1686])                        # J second moment of inertia
    I = 1.0e-12 * np.array([0.0601, 0.3267, 0.8432])                        # I inertia
    G = np.array([2.5091302912e+10, 2.1467424256e+10, 2.9788923392e+10] )   # G torsion constant

    Ux = np.array([21.3, 13.108, 3.5])                  # constant U curvature vectors for each tubes
    Uy = np.array([0, 0, 0])

    ctr = CTRobotModel(no_of_tubes, tubes_length, curve_length, initial_q, E, J, I, G, Ux, Uy)

    ctr_model = lambda q,uz:ctr.moving_CTR(q,uz)
    model = lambda q,uz:UzController(q,uz, dt=Uzdt, model=ctr_model).Uz_controlled_model()

    start_pos = [0, 0, 0.05]
    q_start = np.array([0.0101, 0.0101, 0.0101, 0, 0, 0])  # a_ans, a_ans, a_ans
    uz_0 = np.array([0.0, 0.0, 0.0])
    (r1,r2,r3,Uz) = model(q_start, uz_0)
    start_pos = r1[-1]


    des_pos = [-0.25, -0.25, 0.25]
    hell = HelicalGenerator(start_pos, des_pos)
    ax = plt.axes(projection='3d')
    for xx in np.linspace(0,1,100):
        hell.helical_traj(xx)
        x_2.append(hell.x)
        y_2.append(hell.y)
        z_2.append(hell.z)
        x_v.append(hell.calculate_velocity(0, xx))
        y_v.append(hell.calculate_velocity(1, xx))
        z_v.append(hell.calculate_velocity(2, xx))

    ax.plot3D(x_2, y_2, z_2)
    ax.scatter(x_2[-1], y_2[-1], z_2[-1], label='({:03f},{:03f},{:03f})'.format(x_2[-1], y_2[-1], z_2[-1]))
    ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z ')

    plt.subplots(1)
    tt = np.arange(0, 1, 0.01)
    plt.plot(tt, x_v, label='x')
    plt.plot(tt, y_v, label='y')
    plt.plot(tt, z_v, label='z')
    plt.title('xyz velocity')
    plt.legend()

    plt.show()