from email.base64mime import header_length
import numpy as np
import matplotlib.pyplot as plt
row_vector = np.array([1, 2, 3])
print(row_vector)

column_vector = np.array([[1, 2, 3]]).reshape(-1, 1)
print(column_vector)

# to also make a column vector, newaxis
print(row_vector[: , None])

v = np.array([1, 1])
m = np.array([[1, 2], [1, 2]])
print(v)
print(m)

plt.xlim(0, 3)
plt.ylim(0, 3)

plt.arrow(0, 0, v[0], v[1], head_width=0.1, head_length=0.2)
# plt.show()

print(np.sqrt(np.sum(v ** 2)))

v3dim = np.array([0, 1, 2])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_zlim(0, 10)

ax.quiver(0, 0, 0, *v3dim, arrow_length_ratio=0.1)
plt.show()