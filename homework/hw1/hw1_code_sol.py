
# Hw1 extra credit problem

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

mpl.rcParams['legend.fontsize'] = 10

fig = plt.figure()
ax = fig.gca(projection='3d')
z = np.linspace(0, 20, 100)
x = np.cos(z)
y =np.sin(z)
ax.plot(x, y, z, label='parametric curve')
ax.legend()

plt.show()




fig = plt.figure()
ax = fig.gca(projection='3d')
z = np.linspace(0, 20, 100)
x = np.cos(z)
y =0.1*np.sin(z)+z
ax.plot(x, y, z, label='parametric curve')
ax.legend()

plt.show()



fig = plt.figure()
ax = fig.gca(projection='3d')
z = np.linspace(0, 20, 100)

y =np.sin(z)
x = 0.01*np.cos(z)+y
ax.plot(x, y, z, label='parametric curve')
ax.legend()

plt.show()





