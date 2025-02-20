import Model
import torch
import torch.nn as nn
import dataset3 as dataset
from tqdm import tqdm
import gpu_chooser
import traceback

contextSize = 128
batchSize = 8
epoch = 8192
learning_rate, weight_decay = 5e-3, 1e-5

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

UNSEEN_MASK = torch.ones(batchSize, contextSize, dtype=torch.float32).to(trainingDevice) * -0.1

for n in range(epoch):
    for i in tqdm(range(datar.totalBinSize // (contextSize * batchSize))):
        optim.zero_grad()
        train_sample = datar.makeBatch(batchSize)
        try:
            for view_port_size in range(1, contextSize - 1):
                source = train_sample[:, :view_port_size].to(trainingDevice)
                source = torch.cat([source, UNSEEN_MASK[:, view_port_size:]], dim=1)
                target = train_sample[:, view_port_size:view_port_size + 1].to(trainingDevice)
                #print('Source: {}, Target: {}'.format(source, target))
                modelResponse = theModel(source)
                loss = lossfunc(modelResponse, target)
                print('Loss: {}'.format(loss.item()))
                optim.zero_grad()
                loss.backward()
                optim.step()
        except RuntimeError as e:
            print(traceback.format_exc())
            print('Error at Epoch: {} Batch: {}'.format(n, i))
            if 'illegal memory access' in str(e):
                print('Restarting training')
                exit(-1)
    print('Epoch: {} Loss: {}'.format(n, loss.item()))
    torch.save(theModel.state_dict(), 'model.pth')

while True:
    myStr = input('Enter a string: ')
    while len(myStr) < contextSize:
        inputContext = list(myStr.encode())
        inputContext = torch.tensor(inputContext, dtype=torch.float32).unsqueeze(0).to(trainingDevice) / 2048 + 0.5
        inputContext = torch.cat([inputContext, UNSEEN_MASK[:, inputContext.size(1):]], dim=1)
        modelResponse = theModel(inputContext)
        theWord = chr(((modelResponse[0] - 0.5) * 2048).int())
        if theWord == '\0':
            break
        print(theWord, end='', flush=True)
        myStr += theWord
    print('\n')