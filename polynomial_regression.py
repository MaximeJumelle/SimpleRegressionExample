# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 20:59:37 2017

@author: Maxime
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


def f(x):
    """ function to approximate by polynomial interpolation"""
    return x * np.sin(x)


# generate points used to plot
x_plot = np.linspace(0, 10, 100)

# generate points and keep a subset of them
x = np.linspace(0, 10, 100)
rng = np.random.RandomState(8)
rng.shuffle(x)
x = np.sort(x[:20])
y = f(x)

# create matrix versions of these arrays
X = x[:, np.newaxis]
X_plot = x_plot[:, np.newaxis]

colors = ['teal', 'yellowgreen', 'gold', 'red', 'blue']
lw = 2
plt.plot(x_plot, f(x_plot), color='black', linewidth=lw,
         label="x * sin(x)")
plt.scatter(x, y, color='navy', s=30, marker='o', label="Points")

for count, degree in enumerate([3, 4, 5, 6, 7]):
    model = make_pipeline(PolynomialFeatures(degree), Ridge())
    model.fit(X, y)
    y_plot = model.predict(X_plot)
    plt.plot(x_plot, y_plot, color=colors[count], linewidth=lw,
             label="Degre %d" % degree)
    
plt.legend(loc='lower left')

plt.show()