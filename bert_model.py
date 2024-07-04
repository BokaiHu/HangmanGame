import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import math
import copy

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model, requires_grad=False)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
    

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, num_heads, embedding_dim, dropout):
        super(MultiHeadSelfAttention, self).__init__()
        assert embedding_dim % num_heads == 0
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.q_proj = nn.Linear(in_features=embedding_dim, out_features=embedding_dim, bias=False)
        self.k_proj = nn.Linear(in_features=embedding_dim, out_features=embedding_dim, bias=False)
        self.v_proj = nn.Linear(in_features=embedding_dim, out_features=embedding_dim, bias=False)

        self.dropout = nn.Dropout(p=dropout)

        self.proj = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)

    def forward(self, x, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)

        head_dim = self.embedding_dim // self.num_heads

        n_batches = x.size(0)
        max_sent_length = x.size(1)

        query = self.q_proj(x).view(n_batches, max_sent_length, self.num_heads, head_dim).transpose(1, 2)
        key = self.k_proj(x).view(n_batches, x.size(1), self.num_heads, head_dim).transpose(1, 2)
        value = self.v_proj(x).view(n_batches, x.size(1), self.num_heads, head_dim).transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -np.inf)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        scores = torch.matmul(scores, value)
        scores = scores.transpose(1, 2).contiguous().view(n_batches, max_sent_length, self.num_heads * head_dim)

        return self.proj(scores)
    

class FFN(nn.Module):
    def __init__(self, embedding_dim, mlp_dim, dropout=0.1):
        super(FFN, self).__init__()
        self.w_1 = nn.Linear(embedding_dim, mlp_dim)
        self.w_2 = nn.Linear(mlp_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.gelu(self.w_1(x))))


class EncoderBlock(nn.Module):
    def __init__(self, num_heads, embedding_dim, mlp_dim, dropout):
        super(EncoderBlock, self).__init__()
        self.attn = MultiHeadSelfAttention(num_heads=num_heads, embedding_dim=embedding_dim, dropout=dropout)
        self.ffn = FFN(embedding_dim=embedding_dim, mlp_dim=mlp_dim, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x, mask):
        x = x + self.dropout1(self.attn(self.norm1(x), mask))
        x = x + self.dropout2(self.ffn(self.norm2(x)))
        return x


class BertWithLMHead(nn.Module):
    def __init__(self, vocab_size, embedding_dim, mlp_dim, dropout, device, n_layers=4, num_heads=4):
        super(BertWithLMHead, self).__init__()
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embedding = PositionalEncoding(embedding_dim, dropout)
        self.layers = nn.ModuleList([EncoderBlock(num_heads, embedding_dim, mlp_dim, dropout) for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.generator = LMHead(embedding_dim, vocab_size)
        self.device=device

    def forward(self, x, src_mask):
        x = self.embedding(x) * math.sqrt(self.embedding_dim)
        x = self.pos_embedding(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.layer_norm(x)


class LMHead(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(LMHead, self).__init__()
        self.linear = nn.Linear(in_features=d_model, out_features=vocab_size)

    def forward(self, x, exist_mask):
        result, _ = torch.max(self.linear(x), dim=1)
        result = result.masked_fill_(exist_mask == 1, -1e10)
        return F.log_softmax(result, dim=1)

