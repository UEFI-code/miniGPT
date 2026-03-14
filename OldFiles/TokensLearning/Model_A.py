import torch
import torch.nn as nn
class myBadTransfomerBlock(nn.Module):
    def __init__(self, max_seq_len=128, embdim=64, activation=nn.ReLU):
        super().__init__()
        self.positionEmbedding = nn.Parameter(torch.zeros(1, max_seq_len, embdim), requires_grad=True)
        self.decoder = nn.Sequential(
            nn.Linear(embdim, embdim),
            activation(),
        )

    def forward(self, x):
        # x: [batch, tokens, embdim]
        y = x + self.positionEmbedding[:, :x.size(1)]
        y = y / (y.norm(dim=-1, keepdim=True) + 1e-6)
        cmp_matrix = torch.matmul(y, y.transpose(1, 2))
        y = torch.matmul(cmp_matrix, y) # this step is hybird token's knowledge
        return self.decoder(y)

class myModel(nn.Module):
    def __init__(self, max_seq_len = 128, embeddingDim = 256, num_layers=3):
        super().__init__()
        self.pre_embedding = nn.Embedding(2000, embeddingDim)
        
        self.badtrans = nn.Sequential()
        for _ in range(num_layers):
            self.badtrans.append(myBadTransfomerBlock(embdim=embeddingDim, max_seq_len=max_seq_len))
        self.badtrans_deepth = num_layers
        
        self.windup = nn.Linear(embeddingDim, 2000)

    def forward(self, x, badtrans_now_deepth = None):
        x = self.pre_embedding(x)
        if badtrans_now_deepth is None:
            badtrans_now_deepth = self.badtrans_deepth
        x = self.badtrans[:badtrans_now_deepth](x)
        x = self.windup(x)
        return x

if __name__ == "__main__":
    x = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    badBlocks = myModel()
    print(badBlocks(x))