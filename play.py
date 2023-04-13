import ModelB
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dataset2

class trickObj:
    contextSize = 1024
    str_encoder = dataset2.DataWarpper.str_encoder
    bin_encoder = dataset2.DataWarpper.bin_encoder

myDataPrep = trickObj()
theModel = ModelB.myModel()
# model.pth maybe trained in parallel mode
state_dict = torch.load('model.pth', map_location=torch.device('cpu'))
if 'module' in list(state_dict.keys())[0]:
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    theModel.load_state_dict(new_state_dict)
    print('Model resumed from Parallel checkpoint')
else:
    theModel.load_state_dict(state_dict)
    print('Model resumed from Normal checkpoint')

testStr = 'Hello World'
# testStr = [ord(c) for c in testStr]
# testStr = torch.tensor(testStr, dtype=torch.long)
# testStr = testStr.unsqueeze(0)
exp, testStr = myDataPrep.str_encoder(testStr)
print(exp)
print(testStr)
testStr = torch.tensor(testStr, dtype=torch.long)
testStr = testStr.unsqueeze(0)
respond = theModel(testStr)
respond = np.argmax(respond.detach().numpy(), axis=2)
print(respond)
resBin = b''
for c in respond[0]:
    resBin += int(c).to_bytes(1, 'little')
print(resBin)
try:
    print(resBin.decode('utf-8'))
except:
    print('Decode error')

while True:
    userInput = input('Enter a string: ').encode('utf-8')
    userInput, _ = myDataPrep.bin_encoder(userInput)
    userInput = torch.tensor(userInput, dtype=torch.long)
    userInput = userInput.unsqueeze(0)
    respond = theModel(userInput)
    respond = np.argmax(respond.detach().numpy(), axis=2)
    resBin = b''
    for c in respond[0]:
        resBin += int(c).to_bytes(1, 'little')
    try:
        print(resBin.decode('utf-8'))
    except:
        print('Decode error')