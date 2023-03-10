import ModelA
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dataset

trainingDevice = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

theModel = ModelA.myModel()
contextSize = 1024
optim = torch.optim.Adam(theModel.parameters(), lr=0.0001)
optim.zero_grad()
lossfunc = nn.CrossEntropyLoss()
datar = dataset.DataWarpper(contextSize, './')
batchSize = 16
epoch = 1000

if trainingDevice.type == 'cuda':
    print('Using GPU')
    theModel = theModel.cuda()

for n in range(epoch):
    for _ in range(len(datar.bin) // (contextSize * batchSize)):
        inputContext = datar.makeBatch(batchSize)
        #print(inputContext)
        if trainingDevice.type == 'cuda':
            inputContext = inputContext.cuda()
        modelResponse = theModel(inputContext).permute(0, 2, 1)
        #print(modelResponse.shape)
        # print(inputContext.shape)
        loss = lossfunc(modelResponse, inputContext)
        loss.backward()
        optim.step()
    print('Epoch: {} Loss: {}'.format(n, loss.item()))

torch.save(theModel.state_dict(), 'model.pth')

while True:
    myStr = input('Enter a string: ')
    inputContext = datar.str_encoder(myStr)
    inputContext = torch.tensor(inputContext, dtype=torch.long)
    inputContext = inputContext.unsqueeze(0)
    if trainingDevice.type == 'cuda':
        inputContext = inputContext.cuda()
    modelResponse = theModel(inputContext)
    modelResponse = np.argmax(modelResponse.cpu().detach().numpy(), axis=2)
    resStr = []
    for c in modelResponse[0]:
        if c > 0:
            resStr.append(chr(c))
        else:
            resStr.append('<EOG>')
            break
    print(''.join(resStr))