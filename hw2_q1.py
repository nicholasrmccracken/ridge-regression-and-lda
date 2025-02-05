import numpy as np
from matplotlib import pyplot as plt

# Generating data
np.random.seed(0)
n=10
x = np.linspace(0,3,n)
y = 2.0*x + 1.0+ 0.5*np.random.randn(n)
y[9] = 20

# Least mean squared and ridge regression
X = np.vstack((x, np.ones(n))).T

W_lms = np.linalg.inv(X.T @ X) @ X.T @ y
w_lms, b_lms = W_lms[:-1], W_lms[-1]

lambda_val = 3
I = np.eye(X.shape[1])
W_ridge = np.linalg.inv(X.T @ X + lambda_val * I) @ X.T @ y
w_ridge, b_ridge = W_ridge[:-1], W_ridge[-1]

print(f"LMS: w = {w_lms}, b = {b_lms}")
print(f"Ridge (lambda={lambda_val}): w = {w_ridge}, b = {b_ridge}")

# Plot
plt.plot(x, y, 'o')
plt.plot(x, 2 * x + 1)
plt.plot(x, w_lms * x + b_lms)
plt.plot(x, w_ridge * x + b_ridge)
plt.legend(['data', 'true line', 'lms fitted line', 'ridge fitted line'])
plt.show()
