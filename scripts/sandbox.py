import numpy as np


seed = 2
np.random.seed(seed)

# create 3d array randint
a = np.random.randint(0, 10, (30, 500, 500))


print(a.shape)

smt = a[None]

print(smt.shape)