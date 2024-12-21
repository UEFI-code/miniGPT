import torch
import torch.nn as nn

class myModel(nn.Module):
    def __init__(self, contextSize, emb_dim = 64, deepth = 3, activate_func_cls = nn.ReLU) -> None:
        super().__init__()
        self.encoders = nn.Sequential()
        self.decoders = nn.Sequential()
        for i in range(deepth):
            if i == 0:
                self.encoders.append(nn.Linear(1, emb_dim))
            else:
                self.encoders.append(nn.Linear(emb_dim, emb_dim))
            self.encoders.append(activate_func_cls())
        
        for i in range(deepth):
            if i == deepth - 1:
                self.decoders.append(nn.Linear(emb_dim, 1))
            else:
                self.decoders.append(nn.Linear(emb_dim, emb_dim))
                self.decoders.append(activate_func_cls())

    def forward(self, x):
        x = x.view(x.size(0), x.size(1), 1)
        x = self.encoders(x)
        x = self.decoders(x)
        x = x.view(x.size(0), -1)
        return x

if __name__ == '__main__':
    model = myModel(contextSize=1024)
    print(model(torch.rand(1, 1024)))