import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import exp
from process import get_data

#calculates loss
def loss(T, Y):
	return -T * np.log(Y)

def gradient_w2(X, Y, T):
	return X.T.dot(T - Y)

def gradient_b2(Y, T):
	return np.sum(T - Y)

def feedforward(X, W1, B1, W2, B2):
	Z1 = X.dot(W1) + B1
	A1 = sigmoid(Z1)
	Z2 = A1.dot(W2) + B2
	YP = softmax(Z2)
	return Z1, Z2, YP

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
	print("classification rate: {}".format(class_rate))


X, Y, installs = get_data()

sampleSize = X.shape[0] #sample size
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
Z1, Z2, YP = feedforward(X, W1, B1, W2, B2)

#finds largest index classification
YP = np.argmax(YP, axis = 1)

learning_rate = 1e-7
classRate(YP, Y)

W2 -= learning_rate * gradient_w2(Z2, YP, Y)
B2 -= learning_rate * gradient_b2(YP, Y)
