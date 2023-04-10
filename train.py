import ModelA
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dataset2 as dataset
from tqdm import tqdm

trainingDevice = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

theModel = ModelA.myModel()
contextSize = 1024
optim = torch.optim.Adam(theModel.parameters(), lr=0.0001)
optim.zero_grad()
lossfunc = nn.CrossEntropyLoss()
datar = dataset.DataWarpper(contextSize, './')
batchSize = 8
epoch = 128

if trainingDevice.type == 'cuda':
    print('Using GPU')
    theModel = theModel.cuda()

for n in range(epoch):
    for i in tqdm(range(datar.totalBinSize // (contextSize * batchSize))):
        target, source = datar.makeBatch(batchSize)
        #print(inputContext)
        if trainingDevice.type == 'cuda':
            target = target.cuda()
            source = source.cuda()
        modelResponse = theModel(source).permute(0, 2, 1)
        #print(modelResponse.shape)
        # print(inputContext.shape)
        loss = lossfunc(modelResponse, target)
        loss.backward()
        optim.step()
        if (i + 1) % 1000 == 0:
            print('Epoch: {} Batch: {} Loss: {}'.format(n, i, loss.item()))
            torch.save(theModel.state_dict(), 'model.pth')
    print('Epoch: {} Loss: {}'.format(n, loss.item()))
    torch.save(theModel.state_dict(), 'model.pth')

while True:
    myStr = input('Enter a string: ')
    inputContext, _ = datar.str_encoder(myStr)
    inputContext = torch.tensor(inputContext, dtype=torch.long)
    inputContext = inputContext.unsqueeze(0)
    if trainingDevice.type == 'cuda':
        inputContext = inputContext.cuda()
    rtContext = inputContext
    i = len(myStr)
    for _ in range(i, contextSize):
        modelResponse = theModel(rtContext)
        modelResponse = np.argmax(modelResponse.cpu().detach().numpy(), axis=2)
        if modelResponse[0][i] == 0:
            break
        modelResponse[0][i+1:] = 0
        rtContext = torch.tensor(modelResponse, dtype=torch.long).to(trainingDevice)
        i += 1
    
    resStr = []
    for c in rtContext[0]:
        c = int(c.item())
        if c > 0:
            resStr.append(chr(c))
        else:
            resStr.append('<EOG>')
            break
    print(''.join(resStr))