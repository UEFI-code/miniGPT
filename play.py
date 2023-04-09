import ModelA
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dataset2

class trickObj:
    contextSize = 10
    str_encoder = dataset2.DataWarpper.str_encoder

myDataPrep = trickObj()
theModel = ModelA.myModel()
theModel.load_state_dict(torch.load('model.pth'))

testStr = 'Hello Wor'
# testStr = [ord(c) for c in testStr]
# testStr = torch.tensor(testStr, dtype=torch.long)
# testStr = testStr.unsqueeze(0)
testStr, _ = myDataPrep.str_encoder(testStr)
testStr = torch.tensor(testStr, dtype=torch.long)
testStr = testStr.unsqueeze(0)
print(testStr)
respond = theModel(testStr)
respond = np.argmax(respond.detach().numpy(), axis=2)
print(respond)
resStr = []
for c in respond[0]:
    if c > 0:
        resStr.append(chr(c))
    else:
        resStr.append('<EOG>')
        break

print(''.join(resStr))

while True:
    userInput = input('Enter a string: ')
    userInput, _ = myDataPrep.str_encoder(userInput)
    userInput = torch.tensor(userInput, dtype=torch.long)
    userInput = userInput.unsqueeze(0)
    respond = theModel(userInput)
    respond = np.argmax(respond.detach().numpy(), axis=2)
    resStr = []
    for c in respond[0]:
        if c > 0:
            resStr.append(chr(c))
        else:
            resStr.append('<EOG>')
            break
    print(''.join(resStr))