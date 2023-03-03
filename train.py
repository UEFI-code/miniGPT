import ModelA
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dataset

theModel = ModelA.myModel()
contextSize = 256
optim = torch.optim.Adam(theModel.parameters(), lr=0.0001)
optim.zero_grad()
lossfunc = nn.CrossEntropyLoss()
datar = dataset.DataWarpper(contextSize, 'dataset.py')
 

for _ in range(1000):
    inputContext = datar.makeBatch(10)
    #print(inputContext)
    modelResponse = theModel(inputContext).permute(0, 2, 1)
    # print(modelResponse.shape)
    # print(inputContext.shape)
    loss = lossfunc(modelResponse, inputContext)
    print(loss)
    loss.backward()
    optim.step()

torch.save(theModel.state_dict(), 'model.pth')

while True:
    myStr = input('Enter a string: ')
    inputContext = datar.str_encoder(myStr)
    inputContext = torch.tensor(inputContext, dtype=torch.long)
    inputContext = inputContext.unsqueeze(0)
    modelResponse = theModel(inputContext)
    modelResponse = np.argmax(modelResponse.detach().numpy(), axis=2)
    resStr = []
    for c in modelResponse[0]:
        if c > 0:
            resStr.append(chr(c))
        else:
            resStr.append('<EOG>')
            break
    print(''.join(resStr))