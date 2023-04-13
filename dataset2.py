import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

class DataWarpper():
    def __init__(self, contextSize, folderpath):
        self.contextSize = contextSize
        self.file_index = -1
        self.file_list = []
        self.bin_index = 0
        self.bin = []
        self.totalBinSize = 0
        for i in os.walk(folderpath):
            for j in i[2]:
                if j.endswith('.py') or j.endswith('.txt'):
                    self.file_list.append(os.path.join(i[0], j))
                    self.totalBinSize += os.path.getsize(os.path.join(i[0], j))
        print('Total find files: {}'.format(len(self.file_list)))
        print('Total bin size: {}'.format(self.totalBinSize))
        # shuffle file list
        np.random.shuffle(self.file_list)
    
    def str_encoder(self, theStr):
        Context = []
        for i in range(len(theStr)):
            if i < self.contextSize:
                Context.append(ord(theStr[i]))
            else:
                break
        if len(Context) < self.contextSize:
            Context += [0] * (self.contextSize - len(Context))
        
        Context2 = []
        for i in range(len(theStr) - 1):
            if i < self.contextSize:
                Context2.append(ord(theStr[i]))
            else:
                break
        if len(Context2) < self.contextSize:
            Context2 += [0] * (self.contextSize - len(Context2))
        return Context, Context2
    
    def bin_encoder(self, theBin):
        Context = []
        for i in range(len(theBin)):
            if i < self.contextSize:
                Context.append(theBin[i])
            else:
                break
        if len(Context) < self.contextSize:
            Context += [0] * (self.contextSize - len(Context))
        
        Context2 = []
        for i in range(len(theBin) - 1):
            if i < self.contextSize:
                Context2.append(theBin[i])
            else:
                break
        if len(Context2) < self.contextSize:
            Context2 += [0] * (self.contextSize - len(Context2))
        return Context, Context2
    
    def makeBatch(self, batchSize):
        batch = []
        batch2 = []
        for _ in range(batchSize):
            if len(self.bin) - self.bin_index >= self.contextSize:
                #Buffer Ready
                context, context2 = self.bin_encoder(self.bin[self.bin_index:self.bin_index + self.contextSize])
                batch.append(context)
                batch2.append(context2)
                self.bin_index += self.contextSize
            else:
                if len(self.bin) - self.bin_index > 0:
                    #Read the rest of the buffer, then load new file
                    context, context2 = self.bin_encoder(self.bin[self.bin_index:])
                    batch.append(context)
                    batch2.append(context2)
                    self.bin_index = 0
                    self.file_index += 1
                    self.file_index %= len(self.file_list)
                    print('Loading file: {}'.format(self.file_list[self.file_index]))
                    self.bin = open(self.file_list[self.file_index], 'rb').read()
                else:
                    #index out of buffer, load new file now
                    self.bin_index = 0
                    self.file_index += 1
                    self.file_index %= len(self.file_list)
                    print('Loading file: {}'.format(self.file_list[self.file_index]))
                    self.bin = open(self.file_list[self.file_index], 'rb').read()
                    if len(self.bin) >= self.contextSize:
                        # new file is big enough
                        context, context2 = self.bin_encoder(self.bin[:self.contextSize])
                        batch.append(context)
                        batch2.append(context2)
                    else:
                        # new file is too small
                        context, context2 = self.bin_encoder(self.bin)
                        batch.append(context)
                        batch2.append(context2)
                    self.bin_index += self.contextSize

        return torch.tensor(batch, dtype=torch.long), torch.tensor(batch2, dtype=torch.long)

if __name__ == '__main__':
    dataset = DataWarpper(256, './')
    print(dataset.makeBatch(10))