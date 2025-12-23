import torch
import torch.nn as nn
import math

class myBadTransfomerBlock(nn.Module):
    def __init__(self, embdim=64):
        super().__init__()
        self.phase_A = nn.Linear(embdim, embdim, bias=True)
        self.phase_B = nn.Linear(embdim, embdim, bias=True)
        self.phase_C = nn.Linear(embdim, embdim, bias=True)
        self.out_proj = nn.Linear(embdim, embdim, bias=True)

        self.act = nn.GELU()

        self.norm1 = nn.LayerNorm(embdim)
        self.norm2 = nn.LayerNorm(embdim)

        self.ffn = nn.Sequential(
            nn.Linear(embdim, embdim * 4),
            nn.GELU(),
            nn.Linear(embdim * 4, embdim),
        )

    def forward(self, x):
        y = self.norm1(x)
        A = self.phase_A(y)
        B = self.phase_B(y)
        C = self.phase_C(y)
        attn = torch.matmul(A, B.transpose(1, 2))
        attn = attn / math.sqrt(y.size(-1))
        attn = torch.softmax(attn, dim=-1)
        y = torch.matmul(attn, C)
        y = self.out_proj(y)
        y = y + x
        y = self.norm2(y)
        y = self.ffn(y)
        y = y + x
        return y

class myModel(nn.Module):
    def __init__(self, max_seq_len = 128, embeddingDim = 64, num_layers=3):
        super().__init__()
        self.pre_embedding = nn.Embedding(257, embeddingDim)
        self.positionEmbedding = nn.Parameter(torch.rand(1, max_seq_len, embeddingDim), requires_grad=True)
        
        self.badtrans = nn.Sequential()
        for _ in range(num_layers):
            self.badtrans.append(myBadTransfomerBlock(embdim=embeddingDim))
        self.badtrans_deepth = num_layers
        
        self.final_norm = nn.LayerNorm(embeddingDim)
        self.windup = nn.Linear(embeddingDim, 256)

    def forward(self, x, badtrans_now_deepth = None):
        x = self.pre_embedding(x)
        x = x + self.positionEmbedding[:, :x.size(1)]
        if badtrans_now_deepth is None: badtrans_now_deepth = self.badtrans_deepth
        x = self.badtrans[:badtrans_now_deepth](x)
        x = self.final_norm(x)
        x = self.windup(x)
        return x

if __name__ == "__main__":
    x = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    badBlocks = myModel()
    print(badBlocks(x))