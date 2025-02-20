import torch
import torch.nn as nn
class myBadTransfomerBlock(nn.Module):
    def __init__(self, dim=64, deepth = 2, activation=nn.ReLU(), debug=False):
        super().__init__()
        self.encodingGroup = nn.Sequential()
        self.decodingGroup = nn.Sequential()
        for _ in range(deepth):
            self.encodingGroup.append(nn.Linear(dim, dim, bias=False))
            self.encodingGroup.append(activation)
            self.decodingGroup.append(nn.Linear(dim, dim, bias=False))
            self.decodingGroup.append(activation)
        self.debug = debug
        self.dim = dim

    def forward(self, x):
        y = self.encodingGroup(x) # batch, seq, dim
        y = y / (y.norm(dim=-1, keepdim=True) + 1e-6)
        cmp_matrix = torch.matmul(y, y.transpose(1, 2)) # batch, seq, seq
        if self.debug:
            print(f'Debug: {cmp_matrix}')
        y = torch.matmul(cmp_matrix, y) # this step is hybird token's knowledge
        return self.decodingGroup(y)

class myModel(nn.Module):
    def __init__(self, embeddingDim = 512, embeddingDeepth = 3, num_layers=2, max_seq_len = 128, debug=False):
        super().__init__()
        
        self.pre_embedding = nn.Sequential(
            nn.Linear(1, embeddingDim),
            nn.ReLU(),
            nn.Linear(embeddingDim, embeddingDim),
            nn.ReLU()
        )
        self.positinalEncoding = nn.Parameter(torch.randn(1, max_seq_len, embeddingDim), requires_grad=True)

        self.badtrans = nn.Sequential()
        for _ in range(num_layers):
            self.badtrans.append(myBadTransfomerBlock(dim=embeddingDim, deepth=embeddingDeepth, debug=debug))
        
        self.windup = nn.Sequential(
            nn.Linear(embeddingDim, embeddingDim),
            nn.ReLU(),
            nn.Linear(embeddingDim, 1),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.pre_embedding(x)
        x = x + self.positinalEncoding[:, :x.size(1)]
        x = self.badtrans(x)
        x = self.windup(x)
        return x

if __name__ == "__main__":
    x = torch.randn(1, 4, 64)
    badBlock = myBadTransfomerBlock(debug=True)
    #print(badBlock(x))
    badBlocks = myModel(debug=True)
    x = torch.randn(1, 4, 1)
    print(badBlocks(x))