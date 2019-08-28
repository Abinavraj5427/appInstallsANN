import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import exp

def feedforward(X, W1, B1, W2, B2):
	Z1 = X.dot(W1) + B1
	A1 = sigmoid(Z1)
	Z2 = A1.dot(W2) + B2
	YP = softmax(Z2)
	return YP

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


rawData = pd.read_csv("googleplaystore.csv")
print(rawData)

sampleSize = 100
N = 9
D = 16
M = 8

X = np.random.randn(sampleSize, N)
Y = np.random.randn(sampleSize, M)
Y = np.argmax(Y, axis = 1)

#forward propogation
W1 = np.random.randn(N, D)
B1 = np.random.randn(D)
W2 = np.random.randn(D, M)
B2 = np.random.randn(M)
YP = feedforward(X, W1, B1, W2, B2)

#finds largest index classification
YP = np.argmax(YP, axis = 1)

learning_rate = 1e-7
classRate(YP, Y)
