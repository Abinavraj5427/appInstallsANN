import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import exp
from process import get_data

#calculates loss
def loss(Y, T):
	return (T * np.log(Y)).sum()

def gradient_w1(Y, T, W2, Z, X):
	dZ = (T - Y).dot(W2.T) * Z * (1 - Z)
	ret2 = X.T.dot(dZ)
	return ret2

def gradient_b1(Y, T, W2, Z):
	return ((T - Y).dot(W2.T) * Z * (1-Z)).sum(axis = 0)

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

batchSize = 878
batchX = np.split(X_train, batchSize, axis = 0)
batchY = np.split(Y_train, batchSize, axis = 0)


D = X.shape[1] #features (36)
M = 16 #hidden layers
K = Y.shape[1] #outputs

# X = np.random.randn(sampleSize, N)
# Y = np.random.randn(sampleSize, M)
# Y = np.argmax(Y, axis = 1)

#forward propogation
W1 = np.random.randn(D, M) #(36, 16)
B1 = np.random.randn(M) #(16)
W2 = np.random.randn(M, K) #(16, 19)
B2 = np.random.randn(K) #(19)

losses = []
rates = []
for epoch in range(1000000):
	X = batchX[epoch%len(batchX)]
	Y = batchY[epoch%len(batchY)]

	Z, YP = feedforward(X, W1, B1, W2, B2)

	linear_YP = np.argmax(YP, axis = 1)
	linear_Y = np.argmax(Y, axis = 1)

	l = loss(YP, Y)
	r  =classRate(linear_YP, linear_Y)
	losses.append(l)
	rates.append(r)
	if(epoch%100 == 0):
		print("loss: {} classification rate: {}".format(l, r))

	learning_rate = 1e-4
	W2 += learning_rate * gradient_w2(Z, YP, Y)
	B2 += learning_rate * gradient_b2(YP, Y)
	W1 += learning_rate * gradient_w1(YP, Y, W2, Z, X)
	B1 += learning_rate * gradient_b1(YP, Y, W2, Z)

plt.plot(losses)
plt.show()

plt.plot(rates)
plt.show()