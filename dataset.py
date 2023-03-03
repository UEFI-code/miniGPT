import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

class DataWarpper():
    def __init__(self, contextSize, folderpath):
        self.contextSize = contextSize
        self.index = 0
        self.bin = []
        file_count = 0
        for i in os.walk(folderpath):
            for j in i[2]:
                if j.endswith('.py') or j.endswith('.txt'):
                    self.bin += open(os.path.join(i[0], j), 'rb').read()
                    file_count += 1
        print('Total Loaded Files: {}'.format(file_count))
    
    def str_encoder(self, theStr):
        Context = []
        for i in range(len(theStr)):
            if i < self.contextSize:
                Context.append(ord(theStr[i]))
            else:
                break
        if len(Context) < self.contextSize:
            Context += [0] * (self.contextSize - len(Context))
        return Context
    
    def bin_encoder(self, theBin):
        Context = []
        for i in range(len(theBin)):
            if i < self.contextSize:
                Context.append(theBin[i])
            else:
                break
        if len(Context) < self.contextSize:
            Context += [0] * (self.contextSize - len(Context))
        return Context
    
    def makeBatch(self, batchSize):
        batch = []
        for _ in range(batchSize):
            if self.contextSize <= len(self.bin):
                if self.index + self.contextSize <= len(self.bin):
                    batch.append(self.bin_encoder(self.bin[self.index:self.index + self.contextSize]))
                    self.index += self.contextSize
                else:
                    if self.index < len(self.bin):
                        batch.append(self.bin_encoder(self.bin[self.index:]))
                        self.index = 0
                    else:
                        self.index = 0
                        batch.append(self.str_encoder(self.bin[:self.contextSize]))
                        self.index += self.contextSize
            else:
                batch.append(self.str_encoder(self.bin))
                self.index = 0
        return torch.tensor(batch, dtype=torch.long)

if __name__ == '__main__':
    dataset = DataWarpper(256, './')
    print(dataset.makeBatch(10))