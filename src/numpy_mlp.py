import numpy as np
import pandas as pd
from pathlib import Path

class NeuralNetworkModel:
    #Create a class to store the parameters of the MLP
    def __init__(self, W1, W2, b1, b2):
        self.W1 = W1
        self.W2 = W2
        self.b1 = b1
        self.b2 = b2

def getData():
    #Load the training MNIST data
    ROOT_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = ROOT_DIR / "data"
    df = pd.read_csv(DATA_DIR / "mnist_train.csv").astype(np.float32)
    y_train = df.iloc[:, 0].astype(np.int64).values
    X_train = df.iloc[:, 1:].values/255 #Normalises the input (0-1)
    return X_train, y_train

def initModel():
    he_limit1 = np.sqrt(6/784) #Uniform He initialisation
    W1_init = np.random.uniform(-he_limit1, he_limit1, (784, 64))
    he_limit2 = np.sqrt(6/64)
    W2_init = np.random.uniform(-he_limit2, he_limit2, (64, 10))
    b1_init = np.zeros((1, 64))
    b2_init = np.zeros((1, 10))
    model = NeuralNetworkModel(W1_init, W2_init, b1_init, b2_init)
    return model

def relu(z):
    a = np.maximum(0, z)
    return a

def softMax(z):
    ez = np.exp(z - np.max(z, axis=1, keepdims=True)) #Subtract max value for numerical stability
    a = ez/np.sum(ez, axis=1, keepdims=True)
    return a

def forwardPass(X, model):
    z1 = X @ model.W1 + model.b1
    a1 = relu(z1)
    z2 = a1 @ model.W2 + model.b2
    a2 = softMax(z2)
    return a1, a2, z1 

def crossEntropyLoss(y, a):
    m = y.shape[0]
    eps = 1e-15 #Avoids undefined log(0)
    yPred = np.clip(a, eps, 1 - eps) #Clips data to range to prevent floating point stability issues
    correctClassProb = yPred[np.arange(m), y]
    loss = -np.mean(np.log(correctClassProb))
    return loss

def backwardPass(Xb, yb, a1, a2, z1, model):
    #Compute the gradients of the final layer
    m = Xb.shape[0]
    dz2 = a2.copy()
    dz2[np.arange(m), yb] -= 1
    dz2 /= m #Division by batch size to convert to cost
    dj_db2 = np.sum(dz2, axis=0, keepdims=True)
    dj_dW2 = a1.T @ dz2
    #Compute the gradients of the first layer
    dh1 = dz2 @ model.W2.T
    dz1 = dh1 * (z1 > 0) #Elementwise mask
    dj_db1 = np.sum(dz1, axis=0, keepdims=True)
    dj_dW1 = Xb.T @ dz1
    return dj_dW1, dj_dW2, dj_db1, dj_db2

def updateParams(dj_dW1, dj_dW2, dj_db1, dj_db2, lr, model):
    model.W1 -= lr * dj_dW1
    model.W2 -= lr * dj_dW2
    model.b1 -= lr * dj_db1
    model.b2 -= lr * dj_db2
    return model

def shuffleData(X, y):
    m = X.shape[0]
    shuffleKey = np.random.permutation(m)
    X_shuffled = X[shuffleKey]
    y_shuffled = y[shuffleKey]
    return X_shuffled, y_shuffled

def getBatches(X, y, batchSize):
    m = X.shape[0]
    for start in range(0, m, batchSize):
        Xb = X[start:start+batchSize]
        yb = y[start:start+batchSize]
        yield Xb, yb

def train(model, X_raw, y_raw, epochs, batchSize, lr):
    loss_history = []
    acc_history = []
    for _ in range(epochs):
        X, y = shuffleData(X_raw, y_raw)
        for Xb, yb in getBatches(X, y, batchSize):
            a1, a2, z1  = forwardPass(Xb, model)
            dj_dW1, dj_dW2, dj_db1, dj_db2 = backwardPass(Xb, yb, a1, a2, z1, model)
            model = updateParams(dj_dW1, dj_dW2, dj_db1, dj_db2, lr, model)
        #Store history 
        _, a, _  = forwardPass(X_raw, model)
        epoch_loss = crossEntropyLoss(y_raw, a)
        epoch_acc = accuracy(model, X_raw, y_raw)
        loss_history.append(epoch_loss)
        acc_history.append(epoch_acc)

    historyDict = {
        "loss": loss_history,
        "accuracy": acc_history
    }
    return model, historyDict

def prediction(X, model):
    _, a2, _ = forwardPass(X, model)
    pred = np.argmax(a2, axis=1)
    return pred

def getTest():
    #Load the test MNIST data (same preprocessing as training)
    ROOT_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = ROOT_DIR / "data"
    df = pd.read_csv(DATA_DIR / "mnist_test.csv").astype(np.float32)
    y = df.iloc[:, 0].astype(np.int64).values
    X = df.iloc[:, 1:].values/255
    return X, y

def accuracy(model, X, y):
    #Test the model to find its accuracy
    pred = prediction(X, model)
    accuracy = np.mean(pred == y)*100
    return accuracy

def modelCreation(X_train, y_train, epochs, batchSize, lr):
    #Model creation made faster by not importing the data every time
    model_init = initModel()
    finalModel = train(model_init, X_train, y_train, epochs, batchSize, lr)
    return finalModel