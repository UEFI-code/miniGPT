import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import random

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
    
    def bin_encoder(self, theBin, randomCut=True):
        source = list(theBin)
        if randomCut:
            pos = random.randint(2, len(source)) # leaving 2 bytes as least, one byte as input and another is masked
            source = source[:pos]
        target = source[-1:]
        source[-1] = 0xFF # Mask the target value
        if len(source) < self.contextSize:
            source += [0xFF] * (self.contextSize - len(source))
        return source, target
    
    def bin_encoder_infer(self, theBin):
        source = list(theBin)
        if len(source) < self.contextSize:
            source += [0xFF] * (self.contextSize - len(source))
        return source
    
    def makeBatch(self, batchSize):
        sourceBatch = []
        targetBatch = []
        for _ in range(batchSize):
            if len(self.bin) - self.bin_index >= self.contextSize:
                #Buffer Ready
                source, target = self.bin_encoder(self.bin[self.bin_index:self.bin_index + self.contextSize])
                sourceBatch.append(source)
                targetBatch.append(target)
                self.bin_index += self.contextSize
            else:
                if len(self.bin) - self.bin_index > 0:
                    #Read the rest of the buffer, then load new file
                    source, target = self.bin_encoder(self.bin[self.bin_index:])
                    sourceBatch.append(source)
                    targetBatch.append(target)
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
                        source, target = self.bin_encoder(self.bin[:self.contextSize])
                        sourceBatch.append(source)
                        targetBatch.append(target)
                    else:
                        # new file is too small
                        source, target = self.bin_encoder(self.bin)
                        sourceBatch.append(source)
                        targetBatch.append(target)
                    self.bin_index += self.contextSize

        return torch.tensor(sourceBatch, dtype=torch.float32) / 255, torch.tensor(targetBatch, dtype=torch.float32) / 255

if __name__ == '__main__':
    dataset = DataWarpper(8, './')
    print(dataset.makeBatch(4))