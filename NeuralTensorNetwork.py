import numpy as np
import pandas as pd


class AddLayer:
    def __init__(self):
        pass

    def forward(self, x, y):
        out = x + y

        return out

    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1

        return dx, dy


class DotLayer:
    def __init__(self):
        self.x = None
        self.W = None
        self.dW = None

    def forward(self, x, W):
        self.x = x
        self.W = W
        out = np.dot(x, self.W)

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)

        return dx


class TensorDotLayer:
    def __init__(self):
        self.x = None
        self.W = None
        self.dW = None

    def forward(self, x, W):  # x: 1*500, W: 300*500*500
        self.x = x
        self.W = W

        dotMatrix = np.zeros(500)

        for i in range(len(W)):
            dotValue = np.dot(x, W[i])
            dotMatrix = np.vstack((dotMatrix, dotValue))

        dotMatrix = np.delete(dotMatrix, 0, 0)

        return dotMatrix  # 300*500

    def backward(self, dout):  # dout: 300*500 (k*d)
        dotMatrix = np.zeros((500, 500))

        for i in range(len(dout)):  # len(dout)=300 (k)
            oneDAry = dout[i][None, :]  # 1*500
            dx = np.dot(oneDAry, self.W[i].T)
            dotValue = np.dot(self.x.T, oneDAry)
            dotMatrix = np.dstack((dotMatrix, dotValue))

        dotMatrix = dotMatrix.swapaxes(0, 2).swapaxes(1, 2)
        self.dW = np.delete(dotMatrix, 0, 0)
        # dx = dx[None, :]

        return dx


class HyperbolicTanLayer:
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x

        return np.tanh(x)

    def backward(self, dout):
        dx = dout * (1 - self.x**2)

        return dx
