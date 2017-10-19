# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 21:29:30 2017

@author: Maxime
"""

import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

X = np.sort(20 * np.random.rand(50, 1), axis=0)
y = np.exp(-0.1 * X) * 10 * np.sin(X)
y += 0.5 * np.random.randn(50, 1)

apprRBF = SVR(kernel='rbf', C=1e1, gamma=0.1)
apprPOLY = SVR(kernel='poly', C=1e3, degree=6)

plt.figure(1)

I = np.transpose(np.array([np.linspace(0, 20, 100)]))
y_hat = apprRBF.fit(X, y).predict(I)
y_poly = apprPOLY.fit(X, y).predict(I)

plt.plot(I, np.exp(-0.1*I) * 10 * np.sin(I), color='b', lw=2, label='Theoritical model')
plt.hold('on')

plt.plot(I, y_poly, color='g', lw=2, label='Polynomial')
plt.plot(I, y_hat, color='r', lw=2, label='SVR')
plt.scatter(X, y, color='m', lw=2, label='Data')
plt.legend()
plt.show()