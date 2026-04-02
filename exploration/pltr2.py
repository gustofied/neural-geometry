import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()

ax = fig.add_subplot(111)

x = np.arange(0, 50)
y = np.arange(0, 100)

X, Y = np.meshgrid(x, y)

Z = np.sqrt(X**2 + Y**2)

contour = ax.contourf(X, Y, Z)
plt.colorbar(contour)

plt.show()
