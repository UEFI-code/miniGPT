import Model_A
import torch
import torch.nn as nn
import dataset_A as dataset_A
from tqdm import tqdm
import gpu_chooser
import time

contextSize = 128
batchSize = 1024
epoch = 50000
learning_rate, weight_decay = 1e-4, 0

datar = dataset_A.DataWarpper(contextSize, './demo_txt_dataset')

theModel = Model_A.myModel(max_seq_len=contextSize)

try:
    # model.pth maybe trained in parallel mode
    state_dict = torch.load('model.pth', map_location=torch.device('cpu'))
    print(state_dict.keys())
    badtrans_start_deepth = state_dict['badtrans_now_deepth']
    theModel.load_state_dict(state_dict, strict=False)
    print(f'Model loaded, badtrans_start_deepth: {badtrans_start_deepth}')
    time.sleep(3)
except:
    print('No model checkpoint found, start training from scratch')
    badtrans_start_deepth = 2
    pass

trainingDevice = gpu_chooser.choose_gpu()
theModel = theModel.to(trainingDevice)

def test(test_batch):
    assert test_batch.shape[0] == 1
    def decode_str(x):
        return ''.join([chr(i) for i in x])
    res = ''
    for _ in range(32):
        print(f'In: {test_batch}')
        modelResponse = theModel(test_batch)[0]
        modelResponse = torch.argmax(modelResponse, dim=-1).tolist()
        print(f'Out: {modelResponse}')
        decoded_str = decode_str(modelResponse)
        print(f'Decoded: {decoded_str}')
        res += decoded_str[-1]
        test_batch[0, -1] = modelResponse[-1]
        test_batch = torch.concat((test_batch[:, 1:], torch.tensor([[256]], device=trainingDevice, dtype=torch.long)), dim=1)
    print(f'Final Result: {res}')

optim = torch.optim.SGD(theModel.parameters(), lr=learning_rate, weight_decay=weight_decay)
lossfunc = nn.CrossEntropyLoss()

# input('Press Enter to warm up')

# source, target = datar.makeBatch(batchSize)
# source = source.to(trainingDevice)
# target = target.to(trainingDevice)

# for badtrans_now_deepth in range(1, theModel.badtrans_deepth+1):
#     for i in range(4096):
#         optim.zero_grad()
#         modelResponse = theModel(source, badtrans_now_deepth)
#         loss = lossfunc(modelResponse.view(-1, 256), target.view(-1))
#         print(f'Warmup badtrans_now_deepth {badtrans_now_deepth} epoch {i} Loss: {loss.item()}')
#         loss.backward()
#         optim.step()

# # torch.save(theModel.state_dict(), 'model.pth') # why this trigger a bug?
# test(source[0:1])

# input('Press Enter to start training')

for badtrans_now_deepth in range(badtrans_start_deepth, theModel.badtrans_deepth+1):
    for n in tqdm(range(epoch)):
        for _ in range(1 + datar.totalBinSize // batchSize): # the bin_p shift is equ to batchSize
            source, target = datar.makeBatch(batchSize)
            source = source.to(trainingDevice)
            target = target.to(trainingDevice)
            optim.zero_grad()
            modelResponse = theModel(source, badtrans_now_deepth)
            loss = lossfunc(modelResponse.view(-1, 256), target.view(-1))
            print(f'badtrans_now_deepth {badtrans_now_deepth} epoch {n} Loss: {loss.item()}')
            loss.backward()
            optim.step()
        if n % 512 == 0:
            state_dict = theModel.state_dict()
            state_dict['badtrans_now_deepth'] = badtrans_now_deepth
            torch.save(state_dict, 'model.pth')
            print('Model saved')

test(source[0:1])