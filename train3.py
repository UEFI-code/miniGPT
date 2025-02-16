import Model2 as Model
import torch
import torch.nn as nn
import dataset3 as dataset
from tqdm import tqdm
import gpu_chooser
import time

contextSize = 128
batchSize = 128
epoch = 999990
learning_rate, weight_decay = 5e-6, 1e-5

datar = dataset.DataWarpper(contextSize, './')

trainingDevice = gpu_chooser.choose_gpu()

theModel = Model.myModel()

STAGE = 1

try:
    theModel.encoder.load_state_dict(torch.load('model_encoder.pth'))
    print('Model resumed from encoder checkpoint')
except:
    print('No encoder checkpoint found, start training from scratch')
    pass
try :
    theModel.decoder.load_state_dict(torch.load('model_decoder.pth'))
    print('Model resumed from decoder checkpoint')
except:
    print('No decoder checkpoint found, start training from scratch')
    pass
try:
    theModel.transform_blocks.load_state_dict(torch.load('model_transform_blocks.pth'))
    print('Model resumed from transform_blocks checkpoint')
except:
    print('No transform_blocks checkpoint found, start training from scratch')
    pass

# try:
#     # model.pth maybe trained in parallel mode
#     state_dict = torch.load('model.pth', map_location=torch.device('cpu'))
#     if 'module' in list(state_dict.keys())[0]:
#         new_state_dict = {}
#         for k, v in state_dict.items():
#             name = k[7:]
#             new_state_dict[name] = v
#         theModel.load_state_dict(new_state_dict)
#         print('Model resumed from Parallel checkpoint')
#     else:
#         theModel.load_state_dict(state_dict)
#         print('Model resumed from Normal checkpoint')
# except:
#     print('No model checkpoint found, start training from scratch')
#     pass

time.sleep(3)

theModel = theModel.to(trainingDevice)

if STAGE == 0:
    optim = torch.optim.Adam(list(theModel.encoder.parameters()) + list(theModel.decoder.parameters()), lr=learning_rate, weight_decay=weight_decay)
else:
    optim = torch.optim.Adam(theModel.transform_blocks[:STAGE].parameters(), lr=learning_rate, weight_decay=weight_decay)

lossfunc = nn.L1Loss()

# if trainingDevice.type == 'cuda':
#     theModel = nn.DataParallel(theModel)

for n in range(epoch):
    for i in tqdm(range(datar.totalBinSize // (contextSize * batchSize))):
        optim.zero_grad()
        source, target = datar.makeBatch(batchSize)
        #print(inputContext)
        source = source.to(trainingDevice)
        target = target.to(trainingDevice)
        try:
            modelResponse = theModel(source, STAGE)
            #print(modelResponse)
            #print(modelResponse.shape)
            # print(inputContext.shape)
            loss = lossfunc(modelResponse, target) if STAGE > 8 else lossfunc(modelResponse, source)
            loss.backward()
            optim.step()
        except RuntimeError as e:
            print(e)
            print('Error at Epoch: {} Batch: {}'.format(n, i))
            if 'illegal memory access' in str(e):
                print('Restarting training')
                exit(-1)
    print('Epoch: {} Loss: {}'.format(n, loss.item()))
    if n % 1000 == 999:
        torch.save(theModel.encoder.state_dict(), 'model_encoder.pth')
        torch.save(theModel.decoder.state_dict(), 'model_decoder.pth')
        torch.save(theModel.transform_blocks.state_dict(), 'model_transform_blocks.pth')

while True:
    myStr = input('Enter a string: ')
    while len(myStr) < contextSize:
        inputContext = list(myStr.encode('utf-8'))
        inputContext = torch.tensor(inputContext, dtype=torch.float32).unsqueeze(0).to(trainingDevice) / 255
        modelResponse = theModel(inputContext, STAGE)[0]
        for i in modelResponse:
            print(chr((i * 256).int()), end='', flush=True)
        print()
        theWord = chr((modelResponse[-1] * 256).int())
        if theWord == '\0':
            break
        #print(theWord, end='', flush=True)
        myStr += theWord
    print('\n')