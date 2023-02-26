import torch
import torch.nn as nn
import torch.nn.functional as F

class myModel(nn.Module):
    def __init__(self):
        super(myModel, self).__init__()
        self.Emb = nn.Embedding(95 + 1, 256) # Char level embedding
        self.Conv1 = nn.Conv1d(256, 256, 3, stride=1, padding=1)
        self.Conv2 = nn.Conv1d(256, 256, 3, stride=1, padding=1)
        self.Conv3 = nn.Conv1d(256, 256, 3, stride=1, padding=1)
        self.fc = nn.Linear(256, 95 + 1)
        self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        x = self.Emb(x)
        x = x.permute(0, 2, 1)
        x = F.relu(self.Conv1(x))
        x = F.relu(self.Conv2(x))
        x = F.relu(self.Conv3(x))
        x = x.permute(0, 2, 1)
        x = self.fc(x)
        x = self.softmax(x)
        return x