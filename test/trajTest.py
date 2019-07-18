import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class TrajectoryGenerator():
    def __init__(self, z_max=1):
        self.t = 0
        self.x = 0
        self.y = 0
        self.z = 0
        self.z_max = z_max
        self.r = self.t

    def helical_traj(self, t):
        self.t = t
        self.r = self.t/2
        self.x = self.r*np.cos(self.t)  
        self.y = self.r*np.sin(self.t)
        self.z = self.z_max*self.t

    def helical_getVel(self):
        self.x = self.r*np.cos(self.t)  
        self.y = self.r*np.sin(self.t)
        self.z = self.z_max*self.t


x_2 = []
y_2 = []
z_2 = []
hell = TrajectoryGenerator()
ax = plt.axes(projection='3d')
theta = np.radians(np.linspace(0,360*2,1000))
for xx in theta:
    hell.helical_traj(xx)
    x_2.append(hell.x)
    y_2.append(hell.y)
    z_2.append(hell.z)
ax.plot3D(x_2, y_2, z_2)

plt.show()