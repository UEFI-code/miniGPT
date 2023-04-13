import torch
import torch.nn as nn
import torch.nn.functional as F

class myModel(nn.Module):
    def __init__(self, contextSize=1024):
        super(myModel, self).__init__()
        self.Emb = nn.Embedding(256, 4096) # Char level embedding
        encoders = []
        for _ in range(16):
            encoders.append(nn.Linear(4096, 4096))
        encoders.append(nn.Linear(4096, 256))
        self.encoders = nn.ModuleList(encoders)
        decoders = []
        for _ in range(16):
            decoders.append(nn.Linear(contextSize, contextSize))
        self.decoders = nn.ModuleList(decoders)
    
    def forward(self, x):
        x = self.Emb(x)
        for i in self.encoders:
            x = i(x)
            x = F.relu(x)
        x = x.permute(0, 2, 1)
        for i in self.decoders:
            x = i(x)
            x = F.relu(x)
        x = x.permute(0, 2, 1)
        return x
