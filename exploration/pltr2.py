import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()

ax = fig.add_subplot(111, projection="3d")

x = np.arange(0, 50)
y = np.arange(0, 100)

X, Y = np.meshgrid(x, y)

Z = np.sqrt(X**2 + Y**2)

ax.plot_surface(X, Y, Z)

plt.show()
