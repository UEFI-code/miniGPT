#import ModelB
import BadTransformerLLM
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dataset2 as dataset
from tqdm import tqdm

trainingDevice = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

theModel = BadTransformerLLM.myModel()
try:
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
except:
    print('No model checkpoint found, start training from scratch')
    pass
contextSize = 1024
optim = torch.optim.Adam(theModel.parameters(), lr=0.0001)
optim.zero_grad()
lossfunc = nn.CrossEntropyLoss()
datar = dataset.DataWarpper(contextSize, '/storage/nfs/uefi/miniGPTDataset/')
batchSize = 8
epoch = 128

if trainingDevice.type == 'cuda':
    print('Using GPU')
    theModel = theModel.cuda()
    # Enable Data Parallelism
    theModel = nn.DataParallel(theModel)

for n in range(epoch):
    for i in tqdm(range(datar.totalBinSize // (contextSize * batchSize))):
        optim.zero_grad()
        target, source = datar.makeBatch(batchSize)
        #print(inputContext)
        if trainingDevice.type == 'cuda':
            target = target.cuda()
            source = source.cuda()
        try:
            modelResponse = theModel(source).permute(0, 2, 1)
            #print(modelResponse.shape)
            # print(inputContext.shape)
            loss = lossfunc(modelResponse, target)
            loss.backward()
            optim.step()
        except RuntimeError as e:
            print(e)
            print('Error at Epoch: {} Batch: {}'.format(n, i))
            if 'illegal memory access' in str(e):
                print('Restarting training')
                exit(-1)
        if (i + 1) % 10 == 0:
            print('\nEpoch: {} Batch: {} Loss: {}'.format(n, i, loss.item()))
            torch.save(theModel.state_dict(), 'model.pth')
    print('Epoch: {} Loss: {}'.format(n, loss.item()))
    torch.save(theModel.state_dict(), 'model.pth')
    datar.__init__(contextSize, '/storage/nfs/uefi/miniGPTDataset/')

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
