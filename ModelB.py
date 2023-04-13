import torch
import torch.nn as nn
import torch.nn.functional as F

class myModel(nn.Module):
    def __init__(self, contextSize=1024):
        super(myModel, self).__init__()
        self.Emb = nn.Embedding(256, 4096) # Char level embedding
        self.li1 = nn.Linear(4096, 4096)
        self.li2 = nn.Linear(4096, 4096)
        self.li3 = nn.Linear(4096, 256)
        self.li4 = nn.Linear(contextSize, contextSize)
        self.li5 = nn.Linear(contextSize, contextSize)
        self.li6 = nn.Linear(contextSize, contextSize)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.Emb(x)
        x = self.relu(self.li1(x))
        x = self.relu(self.li2(x))
        x = self.relu(self.li3(x))
        x = x.permute(0, 2, 1)
        x = self.relu(self.li4(x))
        x = self.relu(self.li5(x))
        x = self.relu(self.li6(x))
        x = x.permute(0, 2, 1)
        return x
