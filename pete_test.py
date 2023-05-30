import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append(r'C:\Users\peteb\source\repos\GPy')
import GPy
import random

# Make some 1D training data
N = 500                         # 500 training points
X = np.linspace(0, 10, N)       # Inputs evenly spaced between 0 and 10
F = np.sin(X)                   # True function (f = sin(x))
sigma = 0.1                     # Noise standard deviation
Y = F + sigma * np.random.randn(N)  # Observations

M = 10
indices = random.sample(range(N), M)
Z = X[indices]

# Sort out dimensions
X = np.vstack(X)
Y = np.vstack(Y)
Z = np.vstack(Z)

# GPy
m = GPy.models.SparseGPRegression(X, Y, Z=Z)
m.likelihood.variance = sigma**2
obj_GPy = m._log_marginal_likelihood[0][0] * -1
print(obj_GPy)
