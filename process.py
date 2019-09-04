import pandas as pd
import numpy as np
import numbers

def get_data():
    #Creating my Datasets
    X = pd.read_csv("googleplaystore.csv")

    #removing commas before adding to Y dataset
    X['Installs'] = X['Installs'].str.replace(',', '')
    Y = X['Installs'].values

    #remove uneeded cols
    X = X.drop(columns =['Installs',"App", "Type", "Genres", "Size"])

    #Remove dollar sign from price
    X['Price'] = X['Price'].str.replace('$', '')

    X = X.values

    #remove unclean data of nan ratings
    Y = Y[~pd.isnull(X[:,1])]
    X = X[~pd.isnull(X[:,1]),:]

    #convert strings to floats
    X[:, 2].astype(np.float)
    X[:, 3] = np.asfarray(X[:,3], float)

    categories = np.unique(X[:,0])

    #normalizing ratings, review, and price amt
    X[:, 1] = (X[:,1] - X[:,1].mean())/X[:,1].std()
    X[:, 2] = (X[:,2] - X[:,2].mean())/X[:,2].std()
    X[:, 3] = (X[:,3] - X[:,3].mean())/X[:,3].std()

    #finding all unique outputs
    installs = np.unique(Y)
    installs = np.sort(installs)
    
    #ratings, reviews, and price (D = input features in this case)
    D = 3
    #sample size in this case
    N = X.shape[0]
    catOHE = categories.shape[0]

    #category feature one hot encoding
    X2 = np.zeros((N, D+catOHE))
    X2[:,0:D] = X[:,1:D+1]

    for n in range(N):
        t = categories.tolist().index(X[n,0])
        X2[n,t+D] = 1

    #number of possible outputs
    K = installs.shape[0]

    #mapping outputs to indices
    Y2 = np.zeros((N, K))
    for n in range(N):
        t = installs.tolist().index(Y[n])
        Y2[n, t] = 1

    return X2, Y2, installs