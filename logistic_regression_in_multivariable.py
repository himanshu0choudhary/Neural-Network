import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from sklearn import linear_model

aj = pd.read_csv('Data/age_job.csv')

X_train1 = np.array(aj['age'], dtype=float)
X_train2 = np.array(aj['dis'], dtype=float)
Y_train = np.array(aj['job'], dtype=float)

l_rate = 0.1
n = len(X_train1)
iteration = 10000

m1 = m2 = c = float(0)


def h(x1, x2):
    r = np.array([], dtype=float)
    for i in range(len(x1)):
        r = np.hstack((r, 1 / (1 + math.exp(-m1 * x1[i] - m2 * x2[i] - c))))
    return r


for i in range(iteration):
    s = h(X_train1,X_train2) - Y_train
    t1 = l_rate * np.sum(s) / n
    t2 = l_rate * np.sum(s * X_train1) / n
    t3 = l_rate * np.sum(s * X_train2) / n
    c = c - t1
    m1 = m1 - t2
    m2 = m2 - t3

print(m1, " ", m2, " ", c)

lr = linear_model.LogisticRegression()
lr.fit(aj[['age', 'dis']], aj['job'])



def predict(x1, x2):
    t = 1 / (1 + math.exp(-m1 * x1 - m2 * x2 - c))
    if t<0.5:
        return 0
    else:
        return 1


plt.scatter(X_train1, Y_train)
y_plot = np.array([], dtype=float)
for i in len(X_train1):
    y_plot = np.hstack((y_plot, predict(X_train1(i), X_train2(i))))

plt.plot(X_train1, y_plot)
