import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D

mean = np.array([0, 0, 0])
cov = np.array([[4, -1.2, -2.4], [-1.2, 2, 3.0], [-2.4, 3.0, 10]])
#cov = np.array([[4, 0, 0], [0, 10, 0], [0, 0, 10]])
data = np.random.multivariate_normal(mean, cov, size=200)

print(data.shape)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(data[:,0], data[:,1], data[:,2])
val, vec = np.linalg.eig(cov)
ax.quiver(0, 0, 0, *(np.sqrt(val[0]) * 3 * vec[:, 0]), color='black')
ax.quiver(0, 0, 0, *(np.sqrt(val[1]) * 3 * vec[:, 1]), color='black')
ax.quiver(0, 0, 0, *(np.sqrt(val[2]) * 3 * vec[:, 2]), color='black')
plt.show()