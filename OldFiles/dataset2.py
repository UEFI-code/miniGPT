import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import random

class DataWarpper():
    def __init__(self, contextSize, folderpath, bigRAM = True):
        self.contextSize = contextSize
        self.file_index = -1
        self.file_list = []
        self.bin_p = 0
        self.bin = []
        self.totalBinSize = 0
        self.bigRAM = bigRAM
        for i in os.walk(folderpath):
            for j in i[2]:
                if j.endswith('.py') or j.endswith('.txt'):
                    self.file_list.append(os.path.join(i[0], j))
                    self.totalBinSize += os.path.getsize(os.path.join(i[0], j))
        print('Total find files: {}'.format(len(self.file_list)))
        print('Total bin size: {}'.format(self.totalBinSize))
        # shuffle file list
        np.random.shuffle(self.file_list)
        if self.bigRAM:
            print('Wow Big RAM')
            self.bin_list = []
            for f in self.file_list:
                self.bin_list.append(open(f, 'rb').read())
    
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
            while not len(self.bin) - self.bin_p > 0: # No leaving data in buffer, seek next non-empty file
                self.bin_p = 0
                self.file_index += 1
                self.file_index %= len(self.file_list)
                #print('Loading file: {}'.format(self.file_list[self.file_index]))
                if self.bigRAM:
                    self.bin = self.bin_list[self.file_index]
                else:
                    self.bin = open(self.file_list[self.file_index], 'rb').read()
            source, target = self.bin_encoder(self.bin[self.bin_p:self.bin_p + self.contextSize]) # This will NOT crash even if the file is too small
            sourceBatch.append(source)
            targetBatch.append(target)
            self.bin_p += self.contextSize
            
        return torch.tensor(sourceBatch, dtype=torch.float32) / 2048 + 0.5, torch.tensor(targetBatch, dtype=torch.float32) / 2048 + 0.5

if __name__ == '__main__':
    dataset = DataWarpper(8, './')
    print(dataset.makeBatch(4))