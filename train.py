import ModelA
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

theModel = ModelA.myModel()
contextSize = 256
optim = torch.optim.Adam(theModel.parameters(), lr=0.0001)
optim.zero_grad()
lossfunc = nn.CrossEntropyLoss()

def str_encoder(theStr, contextSize):
    Context = []
    for i in range(len(theStr)):
        if i < contextSize:
            Context.append(ord(theStr[i]) - 32)
        else:
            break
    if len(Context) < contextSize:
        Context += [95] * (contextSize - len(Context))
    return Context

for _ in range(1000):
    myStr = 'Hello World'
    inputContext = str_encoder(myStr, contextSize)
    inputContext = torch.tensor(inputContext, dtype=torch.long)
    inputContext = inputContext.unsqueeze(0)
    modelResponse = theModel(inputContext).permute(0, 2, 1)
    # print(modelResponse.shape)
    # print(inputContext.shape)
    loss = lossfunc(modelResponse, inputContext)
    print(loss)
    loss.backward()
    optim.step()

while True:
    myStr = input('Enter a string: ')
    inputContext = str_encoder(myStr, contextSize)
    inputContext = torch.tensor(inputContext, dtype=torch.long)
    inputContext = inputContext.unsqueeze(0)
    modelResponse = theModel(inputContext)
    modelResponse = np.argmax(modelResponse.detach().numpy(), axis=2)
    resStr = []
    for c in modelResponse[0]:
        if c < 95:
            resStr.append(chr(c + 32))
        else:
            resStr.append('<EOG>')
            break
    print(''.join(resStr))