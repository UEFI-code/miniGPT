import torch
import torch.nn as nn

class myBadTransfomerBlock(nn.Module):
    def __init__(self, embdim=64, activation=nn.ReLU):
        super().__init__()
        self.phase_A = nn.Sequential(
            nn.Linear(embdim, embdim, bias=False),
            activation(),
        )
        self.phase_B = nn.Sequential(
            nn.Linear(embdim, embdim, bias=False),
            activation(),
        )
        self.phase_C = nn.Sequential(
            nn.Linear(embdim, embdim, bias=False),
            activation(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(embdim, embdim, bias=False),
            activation(),
        )

    def forward(self, x):
        #y = y / (y.norm(dim=-1, keepdim=True) + 1e-6)
        A, B, C = self.phase_A(x), self.phase_B(x), self.phase_C(x)
        cmp_matrix = torch.matmul(A, B.transpose(1, 2))
        out = torch.matmul(cmp_matrix, C) # this step is hybird token's knowledge
        return self.decoder(out)

class myModel(nn.Module):
    def __init__(self, max_seq_len = 128, embeddingDim = 64, num_layers=3):
        super().__init__()
        self.pre_embedding = nn.Embedding(257, embeddingDim)
        self.positionEmbedding = nn.Parameter(torch.rand(1, max_seq_len, embeddingDim), requires_grad=True)
        
        self.badtrans = nn.Sequential()
        for _ in range(num_layers):
            self.badtrans.append(myBadTransfomerBlock(embdim=embeddingDim))
        self.badtrans_deepth = num_layers
        
        self.windup = nn.Linear(embeddingDim, 256)

    def forward(self, x, badtrans_now_deepth = None):
        x = self.pre_embedding(x)
        x = x + self.positionEmbedding[:, :x.size(1)]
        if badtrans_now_deepth is None: badtrans_now_deepth = self.badtrans_deepth
        x = self.badtrans[:badtrans_now_deepth](x)
        x = self.windup(x)
        return x

if __name__ == "__main__":
    x = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    badBlocks = myModel()
    print(badBlocks(x))