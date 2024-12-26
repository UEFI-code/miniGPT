import torch
import torch.nn as nn

class BadTransformerBlockHead(nn.Module):
    def __init__(self, emb_dim = 512, deepth = 3, activate_func_cls = nn.PReLU) -> None:
        super().__init__()
        self.encoders = nn.Sequential()
        for i in range(deepth):
            if i == 0:
                self.encoders.append(nn.Linear(1, emb_dim))
            else:
                self.encoders.append(nn.Linear(emb_dim, emb_dim))
            self.encoders.append(activate_func_cls())
    
    def forward(self, x):
        x = x.view(x.size(0), x.size(1), 1)
        x = self.encoders(x)
        #print(x)
        return x
    
class BadTransformerBlockMiddle(nn.Module):
    def __init__(self, emb_dim = 512, deepth = 3, activate_func_cls = nn.PReLU) -> None:
        super().__init__()
        # self.encoders = nn.Sequential()
        self.decoders = nn.Sequential()
        # for _ in range(deepth):
        #     self.encoders.append(nn.Linear(emb_dim, emb_dim))
        #     self.encoders.append(activate_func_cls())
        
        for i in range(deepth):
            if i == 0:
                self.decoders.append(nn.Linear(emb_dim * 2, emb_dim))
            else:
                self.decoders.append(nn.Linear(emb_dim, emb_dim))
            self.decoders.append(activate_func_cls())
        
        # self.badTransformerAdapterA = nn.Linear(emb_dim, emb_dim)
        # self.badTransformerAdapterB = nn.Linear(emb_dim, emb_dim)
        # self.badTransformerAdapterC = nn.Linear(emb_dim, emb_dim)
    
    def forward(self, x):
        # x = self.encoders(x)
        # xA = self.badTransformerAdapterA(x)
        # xB = self.badTransformerAdapterB(x)
        # xC = self.badTransformerAdapterC(x)
        matrix = torch.matmul(x, x.transpose(1, 2)) / x.size(2)
        matrix = torch.matmul(matrix, x) / x.size(1)
        x = self.decoders(torch.cat([x, matrix], dim=2))
        return x

class BadTransformerBlockTail(nn.Module):
    def __init__(self, emb_dim = 512, deepth = 3, activate_func_cls = nn.PReLU) -> None:
        super().__init__()
        self.decoders = nn.Sequential()
        for i in range(deepth):
            if i == deepth - 1:
                self.decoders.append(nn.Linear(emb_dim, 1))
            else:
                self.decoders.append(nn.Linear(emb_dim, emb_dim))
                self.decoders.append(activate_func_cls())
    
    def forward(self, x):
        x = self.decoders(x)
        x = x.view(x.size(0), -1)
        return x

class myModel(nn.Module):
    def __init__(self, deepth = 64) -> None:
        super().__init__()
        self.encoder = BadTransformerBlockHead(emb_dim=1024, activate_func_cls=nn.ReLU)
        self.decoder = BadTransformerBlockTail(emb_dim=1024, activate_func_cls=nn.ReLU)

        self.transform_blocks = nn.Sequential()
        for _ in range(deepth):
            self.transform_blocks.append(BadTransformerBlockMiddle(emb_dim=1024, activate_func_cls=nn.ReLU))
    
    def forward(self, x, stage = 0):
        x = self.encoder(x)
        if stage > 0:
            x = self.transform_blocks[:stage](x)
        x = self.decoder(x)
        return x

if __name__ == '__main__':
    model = myModel()
    # random model parameters
    # for p in model.parameters():
    #     p.data = torch.rand_like(p)
    print(model(torch.rand(1, 16), 4))