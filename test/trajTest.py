import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class HelicalGenerator():
    def __init__(self, total_time, dt, z_max=0.01):
        self.theta = 0
        self.x = 0
        self.y = 0
        self.z = 0
        self.dt = dt
        self.z_max = z_max
        self.r = self.theta
        self.total_time = total_time

    def helical_traj(self, t):
        self.theta = t
        self.r = self.theta/30
        self.x = 1.25 * self.r*np.cos(self.theta)  
        self.y = 1.25 * self.r*np.sin(self.theta)
        self.z = 0.2 + self.z_max*self.theta

    def helical_getVel(self):
        self.x = self.r*np.cos(self.theta)  
        self.y = self.r*np.sin(self.theta)
        self.z = self.z_max*self.theta

    def helical_getTraj(self, t):
        theta = np.radians(np.linspace(180, 180*2.5, int(self.total_time/self.dt)))
        return self.helical_traj(theta[t])


if __name__ == "__main__":
    x_2 = []
    y_2 = []
    z_2 = []

    # hell = TrajectoryGenerator()
    # ax = plt.axes(projection='3d')
    # theta = np.radians(np.linspace(180,180*2.5,1000))
    # for xx in theta:
    #     hell.helical_traj(xx)
    #     x_2.append(hell.x)
    #     y_2.append(hell.y)
    #     z_2.append(hell.z)

    hell = HelicalGenerator(1, 0.001)
    ax = plt.axes(projection='3d')
    for xx in np.arange(1000):
        hell.helical_getTraj(xx)
        x_2.append(hell.x)
        y_2.append(hell.y)
        z_2.append(hell.z)

    ax.plot3D(x_2, y_2, z_2)
    ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z ')

    plt.show()