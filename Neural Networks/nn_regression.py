import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

batch_size = 128
epochs = 100


# Epsilon for loss
epsilon = 0.1

# Polynom coefficients
a = 3
b = 2
c = 0

def decision (x):
    return (a*x**2 + b*x + c) * np.sin(x) / x

def evaluation(y_theo, y):
	if (len(y_theo) != len(y)):
		return 0
	loss = 0
	acc = 0
	for i in range(0,len(y_theo)):
		if (abs(y_theo[i] - y[i]) > epsilon):
			loss += abs(y_theo[i] - y[i])
		else:
			acc += 1.0
	return [loss / len(y_theo), acc / len(y_theo) ]


n = 1000
ntest = 100

flow = np.array(-5 + np.random.rand(n, 1) * 10)
#pdata = pd.DataFrame({"a":flow})  

Xtrain = np.array(np.linspace(-5, 5, n))
Xtest = np.array(np.linspace(-5, 5, ntest))
Ytrain = []
Ytest = []
print Xtrain
for i in range(0,n):
    Ytrain.append(decision(Xtrain[i]))
for i in range(0,ntest):
    Ytest.append(decision(Xtest[i]))
Ytrain = np.array(Ytrain)
Ytest = np.array(Ytest)
print Ytrain


model = Sequential()
model.add(Dense(200, activation='relu', input_dim=1))
model.add(Dense(100, activation='relu'))
model.add(Dense(1))

model.summary()

model.compile(loss='mean_squared_error',
              optimizer='rmsprop',
              metrics=['acc'])

history = model.fit(Xtrain, Ytrain,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(Xtest, Ytest))

predictedModel = model.predict(Xtest)

score = evaluation(Ytest, predictedModel)
print "Test loss:",score[0]
print "Test accuracy:",score[1]

plt.scatter(Xtrain, Ytrain, color='blue', marker='+')
plt.scatter(Xtest, predictedModel, color='red', marker='+')
plt.show()

