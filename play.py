import ModelA
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

theModel = ModelA.myModel()
#theModel.load_state_dict(torch.load('model.pth'))

testStr = 'Hello Wor'
testStr = [ord(c) for c in testStr]
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