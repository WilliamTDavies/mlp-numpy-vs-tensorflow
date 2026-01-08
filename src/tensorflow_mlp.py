import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
from pathlib import Path

def getData():
    #Load the training MNIST data
    ROOT_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = ROOT_DIR / "data"
    df = pd.read_csv(DATA_DIR / "mnist_train.csv").astype(np.float32)
    y_train = df.iloc[:, 0].astype(np.int64).values
    X_train = df.iloc[:, 1:].values/255 #Normalises the input (0-1)
    return X_train, y_train

def modelCreation(X_train, y_train, epochs, batchSize, lr):
    #Simple MLP for MNIST
    model = keras.Sequential([
        layers.Input(shape=(784,)),
        layers.Dense(units=64, activation='relu', kernel_initializer="he_uniform"),
        layers.Dense(units=10, activation='softmax') 
    ])
    model.compile(
        loss = keras.losses.SparseCategoricalCrossentropy(),
        optimizer = keras.optimizers.SGD(learning_rate=lr),
        metrics=["accuracy"]
    )
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batchSize,
        verbose=0
    )
    historyDict = {
        "loss": history.history["loss"],
        "accuracy": history.history["accuracy"]
    }
    return model, historyDict

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
    pred = model.predict(X)
    predIndex = np.argmax(pred, axis=1)
    accuracy = np.mean(predIndex == y)*100
    return accuracy