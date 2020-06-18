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
tensorSize = 500

params = {'T1': np.random.rand(tensorSize, inputSize, inputSize) * np.sqrt(2 / tensorSize),  # d * d * k
          'T2': np.random.rand(tensorSize, inputSize, inputSize) * np.sqrt(2 / tensorSize),  # d * d * k
          'T3': np.random.rand(tensorSize, tensorSize, tensorSize) * np.sqrt(2 / tensorSize),  # k * k * k
          'W1': np.random.rand(tensorSize, 2 * inputSize) * np.sqrt(2 / tensorSize),  # k * 2d
          'W2': np.random.rand(tensorSize, 2 * inputSize) * np.sqrt(2 / tensorSize),  # k * 2d
          'W3': np.random.rand(tensorSize, 2 * tensorSize) * np.sqrt(2 / tensorSize),  # k * 2k
          'b1': np.random.rand(tensorSize) * np.sqrt(2 / tensorSize),  # k
          'b2': np.random.rand(tensorSize) * np.sqrt(2 / tensorSize),  # k
          'b3': np.random.rand(tensorSize) * np.sqrt(2 / tensorSize)}

params['b1'] = params['b1'][:, None]
params['b2'] = params['b2'][:, None]
params['b3'] = params['b3'][:, None]

# %% R1 forward

dotOneLayerR1 = NTN.TensorDotLayer()
dotTwoLayerR1 = NTN.DotLayer()
dotThreeLayerR1 = NTN.DotLayer()
addOneLayerR1 = NTN.AddLayer()
addTwoLayerR1 = NTN.AddLayer()
tanhOneLayerR1 = NTN.HyperbolicTanLayer()

O1T = O1Ary[0][None, :]  # 1*500, O1.T
P = PAry[0][:, None]
O1nP = np.vstack((O1Ary[0][:, None], P))

# forward
OtDotT = dotOneLayerR1.forward(O1T, params['T1'])
DotP = dotTwoLayerR1.forward(OtDotT, P)
W1DotO1nP = dotThreeLayerR1.forward(params['W1'], O1nP)
plusW1 = addOneLayerR1.forward(DotP, W1DotO1nP)
plusb1 = addTwoLayerR1.forward(plusW1, params['b1'])
R1 = tanhOneLayerR1.forward(plusb1)

# %% R2 forward
dotOneLayerR2 = NTN.TensorDotLayer()
dotTwoLayerR2 = NTN.DotLayer()
dotThreeLayerR2 = NTN.DotLayer()
addOneLayerR2 = NTN.AddLayer()
addTwoLayerR2 = NTN.AddLayer()
tanhOneLayerR2 = NTN.HyperbolicTanLayer()

PT = PAry[0][None, :]
O2 = O2Ary[0][:, None]  # 500*1, O2
PnO2 = np.vstack((PAry[0][:, None], O2))

# forward
PtDotT = dotOneLayerR2.forward(PT, params['T2'])
DotO2 = dotTwoLayerR2.forward(PtDotT, O2)
W2DotPnO2 = dotThreeLayerR2.forward(params['W2'], PnO2)
plusW2 = addOneLayerR2.forward(DotO2, W2DotPnO2)
plusb2 = addTwoLayerR2.forward(plusW2, params['b2'])
R2 = tanhOneLayerR2.forward(plusb2)

# %% U forward
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

# %%
