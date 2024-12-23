import Model2 as Model
import torch
import torch.nn as nn
import dataset3 as dataset
from tqdm import tqdm
import gpu_chooser

contextSize = 128
batchSize = 8
epoch = 1024
learning_rate, weight_decay = 0.001, 0

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
        optim.zero_grad()
        source = datar.makeBatch(batchSize)
        #print(inputContext)
        source = source.to(trainingDevice)
        try:
            modelResponse = theModel(source)
            #print(modelResponse.shape)
            # print(inputContext.shape)
            loss = lossfunc(modelResponse, source)
            loss.backward()
            optim.step()
        except RuntimeError as e:
            print(e)
            print('Error at Epoch: {} Batch: {}'.format(n, i))
            if 'illegal memory access' in str(e):
                print('Restarting training')
                exit(-1)
    print('Epoch: {} Loss: {}'.format(n, loss.item()))
    torch.save(theModel.state_dict(), 'model.pth')

while True:
    myStr = input('Enter a string: ')
    while len(myStr) < contextSize:
        inputContext = list(myStr.encode('utf-8'))
        inputContext = torch.tensor(inputContext, dtype=torch.float32).unsqueeze(0).to(trainingDevice) / 255
        modelResponse = theModel(inputContext)[0]
        for i in modelResponse:
            print(chr((i * 255).int()), end='', flush=True)
        print()
        theWord = chr((modelResponse[-1] * 255).int())
        if theWord == '\0':
            break
        #print(theWord, end='', flush=True)
        myStr += theWord
    print('\n')