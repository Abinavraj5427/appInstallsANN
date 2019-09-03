import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import exp
from process import get_data


#calculates loss
def loss(Y, T):
	return (-T * np.log(Y)).sum()

def gradient_w1(Y, T, W2, Z, X):
	dZ = (T - Y).dot(W2.T) * Z * (1 - Z)
	ret2 = X.T.dot(dZ)
	return ret2

def gradient_b1(Y, T, W2, Z1):
	return (W2.T.dot(T - Y) * Z1 * (1-Z)).sum(axis = 0)

def gradient_w2(X, Y, T):
	return (X.T.dot(T - Y)).sum(axis = 0)

def gradient_b2(Y, T):
	return np.sum(T - Y, axis = 0)

def feedforward(X, W1, B1, W2, B2):
	A1 = X.dot(W1) + B1
	Z = sigmoid(A1)
	A2 = Z.dot(W2) + B2
	YP = softmax(A2)
	return Z, YP

def sigmoid(Z):
	return 1/(1+np.exp(-Z))

def softmax(Z):
	return np.exp(Z)/np.exp(Z).sum(axis =1, keepdims =True)

def classRate(YP, Y):
	#classification rate
	correct = 0
	for i in range(len(Y)):
		if(Y[i] == YP[i]):
			correct = correct+1

	class_rate = correct/len(YP)
	return class_rate
	

train_size = .75
X, Y, installs = get_data()

#train and test sets created
X_train = X[:(int)(X.shape[0]*train_size),:]
X_test = X[(int)(X.shape[0]*train_size):,:]
Y_train = Y[:(int)(Y.shape[0]*train_size)]
Y_test = Y[(int)(Y.shape[0]*train_size):]

batchSize = 439
batchX = np.split(X_train, batchSize, axis = 0)
batchY = np.split(Y_train, batchSize, axis = 0)

N = X.shape[1] #features
D = 16 #hidden layers
M = Y.shape[0] #outputs

# X = np.random.randn(sampleSize, N)
# Y = np.random.randn(sampleSize, M)
# Y = np.argmax(Y, axis = 1)

#forward propogation
W1 = np.random.randn(N, D)
B1 = np.random.randn(D)
W2 = np.random.randn(D, M)
B2 = np.random.randn(M)

for epoch in range(batchSize):
	# X = batchX[epoch]
	# Y = batchY[epoch]
	Z, YP = feedforward(X, W1, B1, W2, B2)

	#finds largest index classification
	YP = np.argmax(YP, axis = 1)
	losses = []
	if(epoch%5 == 0):
		l = loss(YP, Y)
		print("classification rate: {}".format(classRate(YP, Y)))
		print("loss: {}".format(l))
		losses.append(l)

	learning_rate = 1e-7 
	print(type(gradient_w1(YP, Y, W2, Z, X).shape))
	print(type(W1))
	W2 += learning_rate * gradient_w2(Z, YP, Y)
	B2 += learning_rate * gradient_b2(YP, Y)
	W1 += learning_rate * gradient_w1(YP, Y, W2, Z, X)
	B1 += learning_rate * gradient_b1(YP, Y, W2, Z)

plt.plot(losses)
plt.show()