import numpy as np
from matplotlib import pyplot as plt

# Generating data
np.random.seed(0)
n=10
m=8
x= np.linspace(0,3,50)
x_train = np.linspace(0,3,n)
y_train = -x_train**2 +2*x_train + 2 +0.5*np.random.randn(n)
x_test = np.linspace(0,3,m)
y_test = -x_test**2 +2*x_test + 2 + 0.5*np.random.randn(m)

# Ridge regression
phi_train = np.column_stack([x_train, x_train**2, x_train**3, x_train**4, np.ones_like(x_train)])
phi_test = np.column_stack([x_test, x_test**2, x_test**3, x_test**4, np.ones_like(x_test)])

lambda_val = 0.1
I = np.eye(phi_train.shape[1])
W_ridge = np.linalg.inv(phi_train.T @ phi_train + lambda_val * I) @ phi_train.T @ y_train
w_ridge, b_ridge = W_ridge[:-1], W_ridge[-1]

print(f"Ridge (lambda={lambda_val}): w = {w_ridge}, b = {b_ridge}")

# Plot
plt.plot(x_train,y_train,'o')
plt.plot(x_test,y_test,'x')
plt.plot(x,-x**2 +2*x + 2)
plt.legend(['training samples','test samples','true line'])
plt.show()

#######

lambdas = np.arange(0.001, 0.101, 0.001)
train_errors = []
test_errors = []
for lambda_val in lambdas:
    I = np.eye(phi_train.shape[1])
    W_ridge = np.linalg.inv(phi_train.T @ phi_train + lambda_val * I) @ phi_train.T @ y_train
    
    y_train_pred = phi_train @ W_ridge
    y_test_pred = phi_test @ W_ridge

    mse_train = np.mean((y_train - y_train_pred) ** 2)
    mse_test = np.mean((y_test - y_test_pred) ** 2)

    train_errors.append(mse_train)
    test_errors.append(mse_test)

optimal_lambda = lambdas[np.argmin(test_errors)]
print(f"Optimal lambda = {optimal_lambda}")

# Plot training error
plt.plot(lambdas, train_errors)
plt.xlabel("Lambda")
plt.ylabel("MSE")
plt.title("Training Error vs. Lambda")
plt.show()

# Plot test error
plt.plot(lambdas, test_errors)
plt.xlabel("Lambda")
plt.ylabel("MSE")
plt.title("Test Error vs. Lambda")
plt.show()
