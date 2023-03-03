import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DataWarpper():
    def __init__(self, contextSize, filepath):
        self.contextSize = contextSize
        self.index = 0
        self.txt = open(filepath, 'r').read()
    
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
    
    def makeBatch(self, batchSize):
        batch = []
        for _ in range(batchSize):
            if self.contextSize <= len(self.txt):
                if self.index + self.contextSize <= len(self.txt):
                    batch.append(self.str_encoder(self.txt[self.index:self.index + self.contextSize]))
                    self.index += self.contextSize
                else:
                    if self.index < len(self.txt):
                        batch.append(self.str_encoder(self.txt[self.index:]))
                        self.index = 0
                    else:
                        self.index = 0
                        batch.append(self.str_encoder(self.txt[:self.contextSize]))
                        self.index += self.contextSize
            else:
                batch.append(self.str_encoder(self.txt))
                self.index = 0
        return torch.tensor(batch, dtype=torch.long)

if __name__ == '__main__':
    dataset = DataWarpper(256, 'dataset.py')
    print(dataset.makeBatch(10))