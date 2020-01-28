import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()
data=pd.read_csv("Data/flower_data.csv")

X_in=np.array([data["sepal_length"]],dtype=float)
X_in=np.vstack((X_in,data["sepal_width"]))
X_in=np.vstack((X_in,data["petal_length"]))
X_in=np.vstack((X_in,data["petal_width"]))

Y_data=le.fit_transform(data["species"]).reshape(1,-1)
Y_in=np.zeros((3,len(Y_data[0])),dtype=float)

for i in range(len(Y_data[0])):
    Y_in[Y_data[0][i]][i]=1

def sigmoid(x):
    r=1/(1+np.exp(-x))
    return r,x

def relu(x):
    return np.maximum(x,0),x

def Params(x):
    param={}
    np.random.seed(1)
    for i in range(1,x.shape[1]):
        param["W"+str(i)]=np.random.randn(x[0][i],x[0][i-1])*0.01
        param["b" + str(i)] = np.zeros((x[0][i], 1),dtype=float)

    return param

def Linear_forward(A,W,b):
    Z = np.dot(W, A) + b
    cache=(A,W,b)
    return Z,cache

def Linear_activation_forward(A_prev,W,b,activation):
    if activation=="sigmoid":
        Z, linear_cache = Linear_forward(A_prev, W, b)
        A,activation_cache=sigmoid(Z)
    elif activation=="relu":
        Z, linear_cache = Linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    cache=(linear_cache,activation_cache)
    return  A,cache

def L_model_forward(X,parameters):
    caches = []
    A = X
    L = len(parameters)//2
    for l in range(1,L):
        A_pre=A
        A,cache=Linear_activation_forward(A_pre,parameters["W"+str(l)], parameters["b"+str(l)], "relu")
        caches.append(cache)

    AL, cache = Linear_activation_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], "sigmoid")
    caches.append(cache)

    return AL,caches


def compute_cost(AL,Y):
    m=Y.shape[1]
    cost = -(Y * np.log(relu(AL)[0]) + (1 - Y) * np.log(relu(1 - AL)[0])) / m
    return cost

def Linear_backward(dZ,cache):
    A_prev,W,b=cache
    m = A_prev.shape[1]

    dW = (np.dot(dZ, A_prev.T)) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(W.T, dZ)

    return dA_prev,dW,db


def relu_backward(dA,activation_cache):
    z=activation_cache>0
    return dA*z

def sigmoid_backward(dA,activation_cache):
    z=sigmoid(activation_cache)
    z=z[0]
    z=z*(1-z)
    return dA*z

def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = Linear_backward(dZ, linear_cache)

    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = Linear_backward(dZ, linear_cache)

    return dA_prev, dW, db

def L_model_backward(AL,Y,caches):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    current_cache = caches[L - 1]
    grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL,
                                                                                                      cache=current_cache,
                                                                                                      activation="sigmoid")
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l+1)], cache=current_cache, activation="relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2

    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]
    return parameters

def NN(X_train,Y_Train,x):
    parameters = Params(x)
    for i in range(100000):
        AL, caches = L_model_forward(X_train, parameters)
        grads = L_model_backward(AL, Y_Train, caches)
        parameters = update_parameters(parameters, grads, 0.01)
    return parameters

layer=np.array([[4,5,4,3]])
parm=NN(X_in,Y_in,layer)

def pred(x):
    r=L_model_forward(x,parm)[0]
    for i in range(len(r[:,0])):
        if r[i]>=0.5:
            r[i]=1
        else:
            r[i]=0
    return r