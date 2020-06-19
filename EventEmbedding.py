import pandas as pd
import numpy as np
import NeuralTensorNetwork as NTN
import requests
import json
from importlib import reload

reload(NTN)

# %% import data

r1 = requests.get('http://120.126.19.107:3000/embed_sub')
r2 = requests.get('http://120.126.19.107:3000/embed_relation')
r3 = requests.get('http://120.126.19.107:3000/embed_obj')

print(r1.status_code)  # 200成功
print(r2.status_code)  # 200成功
print(r3.status_code)  # 200成功

O1Ary = np.array(json.loads(r1.content))  # str 轉乘json
PAry = np.array(json.loads(r2.content))
O2Ary = np.array(json.loads(r3.content))

print(f'O1 shape: {O1Ary.shape}')
print(f'P shape: {PAry.shape}')
print(f'O2 shape: {O2Ary.shape}')

# %% generate O1r
uniqueO1Ary = np.unique(O1Ary, axis=0)
O1rAry = np.zeros(500)

for i in range(len(O1Ary)):
    randNum = np.random.randint(0, len(uniqueO1Ary))
    O1rAry = np.vstack((O1rAry, uniqueO1Ary[randNum]))

O1rAry = np.delete(O1rAry, 0, 0)

# NTN test
print(O1rAry)

# %% init params
inputSize = len(O1Ary[0])  # 1 * 500
tensorSize = 50

params = {'T1': np.random.rand(tensorSize, inputSize, inputSize) * np.sqrt(2 / tensorSize ** 2),  # d * d * k
          'T2': np.random.rand(tensorSize, inputSize, inputSize) * np.sqrt(2 / tensorSize ** 2),  # d * d * k
          'T3': np.random.rand(tensorSize, tensorSize, tensorSize) * np.sqrt(2 / tensorSize ** 2),  # k * k * k
          'W1': np.random.rand(tensorSize, 2 * inputSize) * np.sqrt(2 / tensorSize ** 2),  # k * 2d
          'W2': np.random.rand(tensorSize, 2 * inputSize) * np.sqrt(2 / tensorSize ** 2),  # k * 2d
          'W3': np.random.rand(tensorSize, 2 * tensorSize) * np.sqrt(2 / tensorSize ** 2),  # k * 2k
          'b1': np.random.rand(tensorSize) * np.sqrt(2 / tensorSize ** 2),  # k
          'b2': np.random.rand(tensorSize) * np.sqrt(2 / tensorSize ** 2),  # k
          'b3': np.random.rand(tensorSize) * np.sqrt(2 / tensorSize ** 2)}

params['b1'] = params['b1'][:, None]
params['b2'] = params['b2'][:, None]
params['b3'] = params['b3'][:, None]


# %% R1 forward
def getEventEmbedding(O1EBAry, PEBAry, O2EBAry, i):
    # R1 forward
    dotOneLayerR1 = NTN.TensorDotLayer()
    dotTwoLayerR1 = NTN.DotLayer()
    dotThreeLayerR1 = NTN.DotLayer()
    addOneLayerR1 = NTN.AddLayer()
    addTwoLayerR1 = NTN.AddLayer()
    tanhOneLayerR1 = NTN.HyperbolicTanLayer()

    O1T = O1EBAry[i][None, :]  # 1*500, O1.T
    P = PEBAry[i][:, None]
    O1nP = np.vstack((O1EBAry[i][:, None], P))

    # forward
    OtDotT = dotOneLayerR1.forward(O1T, params['T1'])
    DotP = dotTwoLayerR1.forward(OtDotT, P)
    W1DotO1nP = dotThreeLayerR1.forward(params['W1'], O1nP)
    plusW1 = addOneLayerR1.forward(DotP, W1DotO1nP)
    plusb1 = addTwoLayerR1.forward(plusW1, params['b1'])
    R1 = tanhOneLayerR1.forward(plusb1)

    # R2 forward
    dotOneLayerR2 = NTN.TensorDotLayer()
    dotTwoLayerR2 = NTN.DotLayer()
    dotThreeLayerR2 = NTN.DotLayer()
    addOneLayerR2 = NTN.AddLayer()
    addTwoLayerR2 = NTN.AddLayer()
    tanhOneLayerR2 = NTN.HyperbolicTanLayer()

    PT = PEBAry[i][None, :]
    O2 = O2EBAry[i][:, None]  # 500*1, O2
    PnO2 = np.vstack((PEBAry[i][:, None], O2))

    # forward
    PtDotT = dotOneLayerR2.forward(PT, params['T2'])
    DotO2 = dotTwoLayerR2.forward(PtDotT, O2)
    W2DotPnO2 = dotThreeLayerR2.forward(params['W2'], PnO2)
    plusW2 = addOneLayerR2.forward(DotO2, W2DotPnO2)
    plusb2 = addTwoLayerR2.forward(plusW2, params['b2'])
    R2 = tanhOneLayerR2.forward(plusb2)

    # U forward
    dotOneLayerU = NTN.TensorDotLayer()
    dotTwoLayerU = NTN.DotLayer()
    dotThreeLayerU = NTN.DotLayer()
    addOneLayerU = NTN.AddLayer()
    addTwoLayerU = NTN.AddLayer()
    tanhOneLayerU = NTN.HyperbolicTanLayer()

    R1nR2 = np.vstack((R1, R2))
    # forward
    R1tDotT3 = dotOneLayerU.forward(R1.T, params['T3'])
    DotR2 = dotTwoLayerU.forward(R1tDotT3, R2)
    W3DotR1nR2 = dotThreeLayerU.forward(params['W3'], R1nR2)
    plusW3 = addOneLayerU.forward(DotR2, W3DotR1nR2)
    plusb3 = addTwoLayerU.forward(plusW3, params['b3'])
    U = tanhOneLayerU.forward(plusb3)

    dictR1 = {"dotOneLayer": dotOneLayerR1,
              "dotTwoLayer": dotTwoLayerR1,
              "dotThreeLayer": dotThreeLayerR1,
              "addOneLayer": addOneLayerR1,
              "addTwoLayer": addTwoLayerR1,
              "tanhOneLayer": tanhOneLayerR1}

    dictR2 = {"dotOneLayer": dotOneLayerR2,
              "dotTwoLayer": dotTwoLayerR2,
              "dotThreeLayer": dotThreeLayerR2,
              "addOneLayer": addOneLayerR2,
              "addTwoLayer": addTwoLayerR2,
              "tanhOneLayer": tanhOneLayerR2}

    dictU = {"dotOneLayer": dotOneLayerU,
             "dotTwoLayer": dotTwoLayerU,
             "dotThreeLayer": dotThreeLayerU,
             "addOneLayer": addOneLayerU,
             "addTwoLayer": addTwoLayerU,
             "tanhOneLayer": tanhOneLayerU}

    return U, dictR1, dictR2, dictU


# %%

def BackPorpagation(dictR1, dictR2, dictU, dout, lr=0.00001):
    # U backward
    dPlusb3 = dictU['tanhOneLayer'].backward(dout)
    dPlusW3, db3 = dictU['addTwoLayer'].backward(dPlusb3)
    dDotR2, dW3R1nR2 = dictU['addOneLayer'].backward(dPlusW3)
    dW3, dR1nR2 = dictU['dotThreeLayer'].backward(dW3R1nR2)
    dDotT3, dR2 = dictU['dotTwoLayer'].backward(dDotR2)
    dR1, dT3 = dictU['dotOneLayer'].backward(dDotT3)

    # R2 backward
    dPlusb2 = dictR2['tanhOneLayer'].backward(dR2)
    dPlusW2, db2 = dictR2['addTwoLayer'].backward(dPlusb2)
    dDotO2, dW2PnO2 = dictR2['addOneLayer'].backward(dPlusW2)
    dW2, dPnO2 = dictR2['dotThreeLayer'].backward(dW2PnO2)
    dDotT2, dO2 = dictR2['dotTwoLayer'].backward(dDotO2)
    dP, dT2 = dictR2['dotOneLayer'].backward(dDotT2)

    # R1 backward
    dPlusb1 = dictR1['tanhOneLayer'].backward(dR1.T)
    dPlusW1, db1 = dictR1['addTwoLayer'].backward(dPlusb1)
    dDotP, dW1O1nP = dictR1['addOneLayer'].backward(dPlusW1)
    dW1, dO1nP = dictR1['dotThreeLayer'].backward(dW1O1nP)
    dDotT1, dP = dictR1['dotTwoLayer'].backward(dDotP)
    dO1, dT1 = dictR1['dotOneLayer'].backward(dDotT1)

    # update
    params['T1'] -= lr * dT1
    params['T2'] -= lr * dT2
    params['T3'] -= lr * dT3
    params['W1'] -= lr * dW1
    params['W2'] -= lr * dW2
    params['W3'] -= lr * dW3
    params['b1'] -= lr * db1
    params['b2'] -= lr * db2
    params['b3'] -= lr * db3


# %%

currentIndex = 0
n = 0

while currentIndex <= len(O1Ary)-1:
    U, dictR1G, dictR2G, dictUG = getEventEmbedding(O1Ary, PAry, O2Ary, currentIndex)
    Ur, buf1, buf2, buf3 = getEventEmbedding(O1rAry, PAry, O2Ary, currentIndex)
    margin = 0.5 - np.linalg.norm(U - Ur)
    print(f'\nnp.linalg.norm(U-Ur): {np.linalg.norm(U - Ur)}')
    print(f'max(0, margin): {max(0, margin)}')

    doutG = -0.01 * U  # L2 norm square

    if max(0, margin) > 0 and n < 3:
        n += 1
        print(f'\nCurrent Index: {currentIndex}')
        print(f'N: {n}')

        BackPorpagation(dictR1G, dictR2G, dictUG, doutG, 0.0001)

    else:
        currentIndex += 1
        n = 0

