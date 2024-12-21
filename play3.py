import Model2 as Model
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dataset2


contextSize = 128

theModel = Model.myModel(contextSize=contextSize)

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

testStr = 'Hello World'
testStr = list(testStr.encode())
testStr = torch.tensor(testStr, dtype=torch.float32).unsqueeze(0) / 255
respond = theModel(testStr)[0]
print(respond)
for i in respond:
    print(chr((i * 255).int()), end='')
print('-----------------')

while True:
    myStr = input('Enter a string: ')
    while len(myStr) < contextSize:
        inputContext = list(myStr.encode())
        inputContext = torch.tensor(inputContext, dtype=torch.float32).unsqueeze(0) / 255
        modelResponse = theModel(inputContext)[0]
        theWord = chr((modelResponse[-1] * 255).int())
        if theWord == '\0':
            break
        print(theWord, end='', flush=True)
        myStr += theWord
    print('\n')