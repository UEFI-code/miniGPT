import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import pickle

np.random.seed(233)

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
                if j.endswith('.token.pkl'):
                    self.file_list.append(os.path.join(i[0], j))
                    self.totalBinSize += os.path.getsize(self.file_list[-1])
        print('Total find files: {}'.format(len(self.file_list)))
        # shuffle file list
        np.random.shuffle(self.file_list)
        if self.bigRAM:
            print('Wow Big RAM')
            self.bin_list = []
            for f in self.file_list:
                self.bin_list.append(pickle.load(open(f, 'rb')))
                self.totalBinSize += len(self.bin_list[-1])
    
    def makeBatch(self, batchSize):
        sourceBatch = []
        targetBatch = []
        item_count = 0
        while item_count < batchSize:
            while not len(self.bin) - self.bin_p > 0: # No leaving data in buffer, seek next non-empty file
                self.bin_p = 0
                self.file_index += 1
                self.file_index %= len(self.file_list)
                #print('Loading file: {}'.format(self.file_list[self.file_index]))
                if self.bigRAM:
                    self.bin = self.bin_list[self.file_index]
                else:
                    self.bin = pickle.load(open(self.file_list[self.file_index], 'rb'))
            if self.bin_p + self.contextSize - 1 > len(self.bin) - 1:
                self.bin_p += self.contextSize - 1 # this will trigger next file loading
                continue
            sourceBatch.append(list(self.bin[self.bin_p:self.bin_p + self.contextSize]))
            targetBatch.append(list(self.bin[self.bin_p:self.bin_p + self.contextSize]))
            sourceBatch[-1][-1] = 1 # MASK
            self.bin_p += 1
            item_count += 1
            
        return torch.tensor(sourceBatch, dtype=torch.long), torch.tensor(targetBatch, dtype=torch.long)

if __name__ == '__main__':
    dataset = DataWarpper(8, './demo_token_dataset')
    print(dataset.makeBatch(4))