
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from scipy import stats
from mayavi import mlab

data = pd.read_csv(r'/afs/inf.ed.ac.uk/user/s18/s1854010/Downloads/DONEresultszz2.csv') 

# data.head()

xx = data[data['x'] < 9000]['x']
yy = data[data['y'] < 9000]['y']
zz = data[data['z'] < 9000]['z']

# threedee = plt.figure().gca(projection='3d')
# # df[cols] = df[df[cols] > 0][cols]
# threedee.scatter(xx, yy, zz, marker='x')
# threedee.set_xlabel('x')
# threedee.set_ylabel('y')
# threedee.set_zlabel('z')
# plt.show()


# mu, sigma = 0, 0.1 
# x = 10*np.random.normal(mu, sigma, 5000)
# y = 10*np.random.normal(mu, sigma, 5000)
# z = 10*np.random.normal(mu, sigma, 5000)

xyz = np.vstack([xx,yy,zz])
kde = stats.gaussian_kde(xyz)
density = kde(xyz)

# # Plot scatter with mayavi
# figure = mlab.figure('DensityPlot')
# pts = mlab.points3d(xx, yy, zz, density, scale_mode='none', scale_factor=0.07)
# mlab.axes()
# mlab.show()

# Plot scatter with mayavi
figure = mlab.figure('DensityPlot')
figure.scene.disable_render = True

pts = mlab.points3d(xx, yy, zz, density, scale_mode='none') 
mask = pts.glyph.mask_points
mask.maximum_number_of_points = xx.size
mask.on_ratio = 1
pts.glyph.mask_input_points = True

figure.scene.disable_render = False 
mlab.axes()
mlab.show()