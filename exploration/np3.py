import numpy as np

print(np.power(np.sqrt(13), 2)  == 13) # "False"
print(np.isclose(np.power(np.sqrt(13), 2), 13))  # "True"

# numpy.isclose and numpy.allclose, and numpy.testing.assert_allclose

M1 = np.arange(0, 10).reshape(2, 5)
M2 = np.arange(10, 20).reshape(2, 5)

np.testing.assert_allclose(M1, M2)