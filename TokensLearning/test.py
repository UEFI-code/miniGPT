import Model_A
import torch
import dataset_A as dataset_A
import gpu_chooser
import tokenizer

contextSize = 128
epoch = 500000

datar = dataset_A.DataWarpper(contextSize, './demo_txt_dataset')

theModel = Model_A.myModel(max_seq_len=contextSize)

state_dict = torch.load('model.pth', map_location=torch.device('cpu'))
badtrans_now_deepth = state_dict['badtrans_now_deepth']
theModel.load_state_dict(state_dict, strict=False)
print(f'Model loaded, badtrans_start_deepth: {badtrans_now_deepth}')

device = gpu_chooser.choose_gpu()
theModel = theModel.to(device)

def test(test_batch):
    assert test_batch.shape[0] == 1
    res = ''
    for _ in range(32):
        print(f'In: {test_batch}')
        modelResponse = theModel(test_batch, badtrans_now_deepth)[0]
        modelResponse = torch.argmax(modelResponse, dim=-1).tolist()
        print(f'Out: {modelResponse}')
        decoded_str = tokenizer.decode_back(modelResponse)
        print(f'Decoded: {decoded_str}')
        res += decoded_str[-1]
        test_batch[0, -1] = modelResponse[-1]
        test_batch = torch.concat((test_batch[:, 1:], torch.tensor([[1]], device=device, dtype=torch.long)), dim=1)
    print(f'Final Result: {res}')

source, _ = datar.makeBatch(1)
source = source.to(device)
test(source)