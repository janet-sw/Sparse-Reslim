import numpy as np
import torch
import math
import torch.nn as nn

class SinusoidalEmbeddings(nn.Module):
    def __init__(self, time_steps:int, embed_dim: int):
        super().__init__()
        position = torch.arange(time_steps).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim))
        embeddings = torch.zeros(time_steps, embed_dim, requires_grad=False)
        embeddings[:, 0::2] = torch.sin(position * div)
        embeddings[:, 1::2] = torch.cos(position * div)
        self.embeddings = embeddings

    def forward(self, x, t):
        embeds = self.embeddings[t].to(x.device)
        return embeds

class EmbeddingDenseLayer(nn.Module):
    def __init__(self,
            c_in: int,
            c_out: int,
            dropout_prob: float):
        super().__init__()
        self.linear1 = nn.Linear(c_in,c_out)
        self.linear2 = nn.Linear(c_out,c_out)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)

    #input here is gonna be just [B,C]
    def forward(self, x):
        return(self.linear2(self.dropout(self.relu(self.linear1(x)))))

class TimeEmbeddings(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels//2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x
