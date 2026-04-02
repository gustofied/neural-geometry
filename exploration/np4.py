import numpy as np
import matplotlib.pyplot as plt

v1 = np.array([1, 1])
v2 = v1 * 2

fig = plt.figure()
ax = fig.add_subplot()

ax.set_xlim(0, 4)
ax.set_ylim(0, 4)

ax.arrow(0, 0, v1[0], v1[1])
ax.arrow(0, 0, v2[0], v2[1])

plt.show()