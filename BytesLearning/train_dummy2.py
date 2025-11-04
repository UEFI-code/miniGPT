import Model_A
import torch
import torch.nn as nn
import dataset_A as dataset_A
from tqdm import tqdm
import gpu_chooser
import time

contextSize = 64
batchSize = 64
epoch = 10000
learning_rate, weight_decay = 1e-3, 0

datar = dataset_A.DataWarpper(contextSize, './demo_txt_dataset')

theModel = Model_A.myModel(max_seq_len=contextSize, num_layers=1)

try:
    # model.pth maybe trained in parallel mode
    state_dict = torch.load('model_dummy.pth', map_location=torch.device('cpu'))
    print(state_dict.keys())
    theModel.load_state_dict(state_dict, strict=False)
    time.sleep(3)
except:
    print('No model checkpoint found, start training from scratch')
    pass

trainingDevice = gpu_chooser.choose_gpu()
print(f'Training on device: {trainingDevice}')
theModel = theModel.to(trainingDevice)

def test(test_batch):
    assert test_batch.shape[0] == 1
    def decode_str(x):
        return ''.join([chr(i) for i in x])
    res = ''
    for _ in range(64):
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

source, target = datar.makeBatch(batchSize)
source = source.to(trainingDevice)
target = target.to(trainingDevice)

for n in tqdm(range(epoch)):
    optim.zero_grad()
    modelResponse = theModel(source)
    loss = lossfunc(modelResponse.view(-1, 256), target.view(-1))
    print(f'Epoch {n} Loss: {loss.item()}')
    loss.backward()
    optim.step()
    if n % 512 == 15:
        state_dict = theModel.state_dict()
        torch.save(state_dict, 'model_dummy.pth')
        print('Model saved')

test(source[0:1])