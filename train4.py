import Model
import torch
import torch.nn as nn
import dataset4 as dataset
from tqdm import tqdm
import gpu_chooser
import traceback

contextSize = 128
batchSize = 8
epoch = 8192
learning_rate, weight_decay = 5e-4, 1e-5

datar = dataset.DataWarpper(contextSize, './')

trainingDevice = gpu_chooser.choose_gpu()

theModel = Model.myModel(contextSize)

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

theModel = theModel.to(trainingDevice)

optim = torch.optim.Adam(theModel.parameters(), lr=learning_rate, weight_decay=weight_decay)

lossfunc = nn.L1Loss()

if trainingDevice.type == 'cuda':
    theModel = nn.DataParallel(theModel)

for n in range(epoch):
    for i in tqdm(range(datar.totalBinSize // (contextSize * batchSize))):
        source, target = datar.makeBatch(batchSize)
        source = source.to(trainingDevice)
        target = target.to(trainingDevice)
        modelResponse = theModel(source)
        loss = lossfunc(modelResponse, target)
        print('Loss: {}'.format(loss.item()))
        optim.zero_grad()
        loss.backward()
        optim.step()
        
    print('Epoch: {} Loss: {}'.format(n, loss.item()))
    torch.save(theModel.state_dict(), 'model.pth')