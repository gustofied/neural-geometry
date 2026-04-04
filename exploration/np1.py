import numpy as np


M = np.array([[10, 20, 30],
              [40, 50, 60],
              [70, 80, 90]])

print("Matrix:")
print(M)

print("\nm[[0, 2]] ->")
print(M)                  

print("\nm[:, [0, 2]] ->")
print(M[:, [0, 2]])           

print("\nm[[0, 2], [1, 0]] ->", M[[0, 2], [1, 0]])  

mask = M > 40
print("\nm[m > 40] ->", M[mask])   

# meh why not
print("--")
print(M[[2,2], [1, 0]])