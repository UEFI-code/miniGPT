import Model2 as Model
import torch
import torch.nn as nn
import torch.nn.functional as F
contextSize = 128

STAGE = 1

theModel = Model.myModel()

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

testStr = 'Hello World'
testStr = list(testStr.encode())
testStr[-1] = -128
testStr = torch.tensor(testStr, dtype=torch.float32).unsqueeze(0) / 255
print(testStr)
respond = theModel(testStr, STAGE)[0]
print(respond)
for i in respond:
    print(chr((i * 256).int()), end='')
print('-----------------')

while True:
    myStr = input('Enter a string: ')
    while len(myStr) < contextSize:
        inputContext = list(myStr.encode())
        inputContext[-1] = -128
        inputContext = torch.tensor(inputContext, dtype=torch.float32).unsqueeze(0) / 255
        modelResponse = theModel(inputContext, STAGE)[0]
        theWord = chr((modelResponse[-1] * 256).int())
        if theWord == '\0':
            break
        print(theWord, end='', flush=True)
        myStr += theWord
    print('\n')