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
        cmp_matrix = torch.matmul(y, y.transpose(1, 2))
        y = torch.matmul(cmp_matrix, y) # this step is hybird token's knowledge
        y = y / y.size(1)
        return self.decoder(y)

class myModel(nn.Module):
    def __init__(self, max_seq_len = 128, embeddingDim = 64, num_layers=2):
        super().__init__()
        self.pre_embedding = nn.Embedding(257, embeddingDim)
        
        self.badtrans = nn.Sequential()
        for _ in range(num_layers):
            self.badtrans.append(myBadTransfomerBlock(embdim=embeddingDim, max_seq_len=max_seq_len))
        
        self.windup = nn.Linear(embeddingDim, 256)

    def forward(self, x):
        x = self.pre_embedding(x)
        x = self.badtrans(x)
        x = self.windup(x)
        return x

if __name__ == "__main__":
    x = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    badBlocks = myModel()
    print(badBlocks(x))