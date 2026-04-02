import numpy as np

print(np.power(np.sqrt(13), 2)  == 13) # "False"
print(np.isclose(np.power(np.sqrt(13), 2), 13))  # "True"

# numpy.isclode and numpy.allclose, and numpy.testing.assert_allclose