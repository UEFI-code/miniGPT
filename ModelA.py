import torch
import torch.nn as nn
import torch.nn.functional as F

class myModel(nn.Module):
    def __init__(self):
        super(myModel, self).__init__()
        self.Emb = nn.Embedding(256, 4096) # Char level embedding
        self.Conv1 = nn.Conv1d(4096, 4096, 3, stride=1, padding=1)
        self.Conv2 = nn.Conv1d(4096, 4096, 3, stride=1, padding=1)
        self.Conv3 = nn.Conv1d(4096, 4096, 3, stride=1, padding=1)
        self.fc1 = nn.Linear(4096, 4096)
        self.fc2 = nn.Linear(4096, 256)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.Emb(x)
        x = x.permute(0, 2, 1)
        x = F.relu(self.Conv1(x))
        x = F.relu(self.Conv2(x))
        x = F.relu(self.Conv3(x))
        x = x.permute(0, 2, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
