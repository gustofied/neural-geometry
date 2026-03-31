import matplotlib.pyplot as plt
import numpy as np

plt.style.use('_mpl-gallery-nogrid')

# make data 
X, Y = np.meshgrid(np.linspace(-3, 3, 256), np.linspace(-3,3, 256))
Z = (1 - X/2 + X**5 +Y**3) * np.exp(-X**2 - Y**2)
levels = np.linspace(np.min(X), np.max(X), 7)

# plot
fig, ax = plt.subplots()

ax.contour(X, Y, Z, levels=levels)

plt.show()


x_values =[1, 2, 3, 4, 5]
y_values = [2, 4, 56, 5, 5 ]

plt.plot(x_values, y_values)

plt.show()


x = np.linspace(0, 5, 11)
y = x ** 2
fig = plt.figure()

axes1 = fig.add_axes((0.1, 0.1, 0.8, 0.8)) # main axes
axes2 = fig.add_axes((0.2, 0.5, 0.4, 0.3)) # inset axes

# Larger Figure Axes 1
axes1.plot(x, y, 'b')
axes1.set_xlabel('X_label_axes2')
axes1.set_ylabel('Y_label_axes2')
axes1.set_title('Axes 2 Title')

# Insert Figure Axes 2
axes2.plot(y, x, 'r')
axes2.set_xlabel('X_label_axes2')
axes2.set_ylabel('Y_label_axes2')
axes2.set_title('Axes 2 Title')

plt.show()