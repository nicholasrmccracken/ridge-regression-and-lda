import numpy as np
import urllib.request
from sklearn.model_selection import train_test_split
from scipy.stats import multivariate_normal

# Load Spambase data
url =" http://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
raw_data = urllib.request.urlopen(url)
dataset = np.loadtxt(raw_data,delimiter=",")
x = dataset[:,0:-1]
y = dataset[:,-1]
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size = 0.30, random_state= 17)

# LDA
n = len(y_train)
mu1 = np.mean(x_train[y_train == 1, :], axis=0)
mu2 = np.mean(x_train[y_train == 0, :], axis=0)
sigma = np.zeros((x_train.shape[1], x_train.shape[1]))
for i in range(n):
    if y_train[i] == 1:
        sigma += np.outer((x_train[i, :] - mu1), (x_train[i, :] - mu1)) / n
    else:
        sigma += np.outer((x_train[i, :] - mu2), (x_train[i, :] - mu2)) / n

# Define Gaussian distributions
g1 = multivariate_normal(mu1, sigma)
g2 = multivariate_normal(mu2, sigma)

# Priors
q1 = np.mean(y_train == 1)
q2 = np.mean(y_train == 0)

# Predict labels for test data
predicted_labels = []
for i in range(len(x_test)):
    if g1.pdf(x_test[i, :]) * q1 >= g2.pdf(x_test[i, :]) * q2:
        predicted_labels.append(1)
    else:
        predicted_labels.append(0)

# Calculate test error
predicted_labels = np.array(predicted_labels)
test_error = np.mean(predicted_labels != y_test)
print(f"Test Error: {test_error}")
