import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.optimize as opt
import pandas as pd
import matplotlib.cm as cm
import matplotlib.mlab as mlab
def plotDecisionBoundary(theta, X, y):
    plt.figure()
    plt.plot(X.ix[y==1,0],X.ix[y==1,1],'o', label = 'Admitted')
    plt.plot(X.ix[y == 0,0],X.ix[y == 0,1],'x',label = 'Not admitted')
    matplotlib.rcParams['xtick.direction'] = 'out'
    matplotlib.rcParams['ytick.direction'] = 'out'

    delta = 0.05
    x1 = np.arange(-1,1.5,delta)
    x2 = np.arange(-1,1.5,delta)
    m = x1.shape[0]
    n = x2. shape[0]
    X = []
    for i in np.arange(m):
        for j in np.arange(n):
            X.append([x1[i],x2[j]])
    X = pd.DataFrame(X)
    z = np.zeros(n * m)
    z = mapFeature(X.ix[:,0],X.ix[:,1]).dot(theta).reshape((n,m))
    print(z.shape)
    x1,x2 = np.meshgrid(x1, x2)
    CS = plt.contour(x1, x2, z,[0])
    plt.clabel(CS, inline = 1, fontsize = 10)
    plt.title('simple')
    plt.show()
def sigmoid(z):
    return 1 / (np.exp(-z) + 1)
def plotData(X,y):
    matplotlib.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots(1)
    ax.plot(X.ix[y==1,0],X.ix[y==1,1],'o', label = 'y = 1')
    ax.plot(X.ix[y == 0,0],X.ix[y == 0,1],'x',label = 'y = 0')
    ax.set_xlabel('Microchip Test 1')
    ax.set_ylabel('Microchip Test 2')
    ax.legend()
    plt.show()
def mapFeature(X1, X2):
    degree = 6
    out = pd.DataFrame(np.ones(X1.shape[0]))
    for i in range(1,degree+ 1):
        for j in range(i+1):
            out = pd.concat([out,X1**(i - j) * X2**j],axis = 1, ignore_index = True)
    return out
def costFunction(theta, X, y, lambdas):
    m = y.size
    J = -1 / m * sum(np.log(sigmoid(X.dot(theta))) * y + np.log(1 - sigmoid(X.dot(theta))) * (1-y)) + lambdas / 2 / m * sum((theta[1:]**2))
#    grad = 1 / m * X.T.dot((sigmoid(X.dot(theta)) - y))
    return J
def gradient(theta, X, y, lambdas):
    m = y.size
    grad = np.zeros_like(theta)
    grad[0] = 1 / m * sum((sigmoid(X.dot(theta)) - y) * X.ix[:,0])
    grad[1:] = 1 / m * (X.ix[:,1:].T).dot(sigmoid(X.dot(theta)) - y) + lambdas / m * theta[1:]
    return grad
def predict(theta, X):
    probalility = sigmoid(X.dot(theta))
    return [1 if i >= 0.5 else 0 for i in probalility]
data = pd.read_csv('ex2data2.txt',delimiter = ',', header = None)

X = data.ix[:,0:2]
y = data.ix[:,2]
#plotData(X,y)
X = mapFeature(X.ix[:,0],X.ix[:,1])

initial_theta = np.zeros(X.shape[1])
lambdas = 1
#cost = costFunction(initial_theta, X, y, lambdas)
#grad = gradient(initial_theta, X, y, lambdas)

result = opt.fmin_tnc(func=costFunction, x0=initial_theta, fprime=gradient, args=(X,y, lambdas))
theta = result[0]
prodictions = predict(theta, X)
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a,b) in zip(prodictions, y) ]

print(sum(correct) / len(correct))



X = data.ix[:,0:2]
y = data.ix[:,2]
plotDecisionBoundary(theta,X,y)