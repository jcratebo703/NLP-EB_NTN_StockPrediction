import numpy as np
import tensorflow as tf
import pandas as pd


class NTN:
    def __init__(self, O1, P, O2):
        print("initializing...")
        self.O1 = O1
        self.P = P
        self.O2 = O2

        self.inputSize = len(O1)  # 1 * 25
        self.tensorSize = len(O1)

        self.params = {'T1': np.random.rand(self.inputSize, self.inputSize, self.tensorSize) / np.sqrt(self.tensorSize),
                       # d * d * k
                       'T2': np.random.rand(self.inputSize, self.inputSize, self.tensorSize) / np.sqrt(self.tensorSize),
                       # d * d * k
                       'T3': np.random.rand(self.tensorSize, self.tensorSize, self.tensorSize) / np.sqrt(
                           self.tensorSize),  # k * k * k
                       'W1': np.random.rand(self.tensorSize, 2 * self.inputSize) / np.sqrt(self.tensorSize),  # k * 2d
                       'W2': np.random.rand(self.tensorSize, 2 * self.inputSize) / np.sqrt(self.tensorSize),  # k * 2d
                       'W3': np.random.rand(self.tensorSize, 2 * self.tensorSize) / np.sqrt(self.tensorSize),  # k * 2k
                       'b1': np.random.rand(self.tensorSize) / np.sqrt(self.tensorSize),  # k
                       'b2': np.random.rand(self.tensorSize) / np.sqrt(self.tensorSize),  # k
                       'b3': np.random.rand(self.tensorSize) / np.sqrt(self.tensorSize)}  # k

    def updateParam(self, T1, T2, T3, W, b):
        self.params['T1'] = T1
        self.params['T2'] = T2
        self.params['T3'] = T3
        self.params['W'] = W
        self.params['b'] = b

        return None

    @staticmethod
    def tensorProduct(A1, T, A2, W, b):
        print("Tensor Product...")
        r1 = np.tensordot(A1, T, axes=1)
        c1 = np.tensordot(r1.T, A2, axes=1)
        c2 = np.tensordot(W, np.append(A1, A2), axes=1)

        return np.tanh(c1 + c2 + b)

    def getEventEmb(self):
        r1 = self.tensorProduct(self.O1, self.params['T1'], self.P, self.params['W1'], self.params['b1'])
        r2 = self.tensorProduct(self.P, self.params['T2'], self.O2, self.params['W1'], self.params['b1'])
        U = self.tensorProduct(r1, self.params['T3'], r2, self.params['W1'], self.params['b1'])

        return U

    def getMarginLoss(self, U, Ur, lamb=0.0001):
        normSqrT1 = np.square(tf.norm(self.params['T1']))
        normSqrT2 = np.square(tf.norm(self.params['T2']))
        normSqrT3 = np.square(tf.norm(self.params['T3']))
        normSqrW = np.square(tf.norm(self.params['W1']))
        normSqrB = np.square(tf.norm(self.params['b1']))

        return max(0, 1)
