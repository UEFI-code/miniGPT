import Model
import torch
import torch.nn as nn
import dataset as dataset
from tqdm import tqdm
import gpu_chooser
import time

contextSize = 128
batchSize = 256
epoch = 8192
learning_rate, weight_decay = 5e-4, 1e-5

datar = dataset.DataWarpper(contextSize, './')

trainingDevice = gpu_chooser.choose_gpu()

theModel = Model.myModel(max_seq_len=contextSize, debug=False)

try:
    # model.pth maybe trained in parallel mode
    state_dict = torch.load('model.pth', map_location=torch.device('cpu'))
    print(state_dict.keys())
    theModel.load_state_dict(state_dict)
    print('Model loaded')
except:
    print('No model checkpoint found, start training from scratch')
    pass

theModel = theModel.to(trainingDevice)

optim = torch.optim.SGD(theModel.parameters(), lr=learning_rate, weight_decay=weight_decay)

time.sleep(5)

lossfunc = nn.L1Loss()

for n in tqdm(range(epoch)):
    for i in range(datar.totalBinSize // (contextSize * batchSize)):
        source, target = datar.makeBatch(batchSize)
        source = source.to(trainingDevice).view(source.size(0), source.size(1), 1)
        target = target.to(trainingDevice)
        optim.zero_grad()
        modelResponse = theModel(source)[:, -1, 0]
        loss = lossfunc(modelResponse, target)
        print('Loss: {}'.format(loss.item()))
        loss.backward()
        optim.step()
    if n % 512 == 0:
        print('Epoch: {} Loss: {}'.format(n, loss.item()))
        torch.save(theModel.state_dict(), 'model.pth')
        print('Model saved')

test_batch, _ = datar.makeBatch(1)
test_batch = test_batch.to(trainingDevice).view(test_batch.size(0), test_batch.size(1), 1)
while True:
    modelResponse = theModel(test_batch)[:, -1, 0]
    theWord = chr(int(modelResponse[-1] * 256))
    if theWord == '\0':
        break
    print(theWord, end='', flush=True)
    test_batch[:, -1, 0] = modelResponse
    test_batch = torch.concat((test_batch[:, 1:], torch.tensor([[[-128 / 255]]], device=trainingDevice, dtype=torch.float32)), dim=1)