import Model_A
import torch
import dataset_A as dataset_A
import gpu_chooser

contextSize = 128
epoch = 500000

datar = dataset_A.DataWarpper(contextSize, './demo_txt_dataset')

theModel = Model_A.myModel(max_seq_len=contextSize)

state_dict = torch.load('model.pth', map_location=torch.device('cpu'))
theModel.load_state_dict(state_dict)
print('Model loaded')

device = gpu_chooser.choose_gpu()
theModel = theModel.to(device)

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
        test_batch = torch.concat((test_batch[:, 1:], torch.tensor([[256]], device=device, dtype=torch.long)), dim=1)
    print(f'Final Result: {res}')

source, _ = datar.makeBatch(1)
source = source.to(device)
test(source)