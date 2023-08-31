import os
import copy
import math
import torch
import torch.nn as nn
from torch.nn import functional as F


def attention(query, key, value):
    """ 
    scaled dot-product attention
        query: [batch, seq, dim]
        key:   [batch, seq, dim]
        value: [batch, seq, dim]
    """
    
    d = query.size(-1)
    score = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d)  # [batch, seq, seq]
    attn = F.softmax(score, dim=-1)
    output = torch.matmul(attn, value)  # [batch, seq, dim]
    return output


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadAttention(nn.Module):
    def __init__(self, head_num, dim):
        super(MultiHeadAttention, self).__init__()
        assert dim % head_num == 0
        self.head_num = head_num
        self.dim = dim

        self.hid_dim = int(self.dim / head_num)
        self.linears = clones(nn.Linear(self.dim, self.dim), 4)

    def forward(self, query, key, value):
        """
            query: [batch, seq, dim]
            key:   [batch, seq, dim]
            value: [batch, seq, dim]
        """
        batch_size = query.size(0)

        query = self.linears[0](query).view(batch_size, -1, self.head_num, self.hid_dim).transpose(1, 2)
        key = self.linears[1](key).view(batch_size, -1, self.head_num, self.hid_dim).transpose(1, 2)
        value = self.linears[2](value).view(batch_size, -1, self.head_num, self.hid_dim).transpose(1, 2)

        output = attention(query, key, value)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.head_num * self.hid_dim)
        return self.linears[3](output)


if __name__ == '__main__':
    torch.manual_seed(1337)
    h = 8
    d_model = 512
    batch_size = 1
    seq_length = 10

    query = torch.randn([batch_size, seq_length, d_model]).cuda()
    key = query.clone()
    value = query.clone()

    model1 = MultiHeadAttention(h, d_model).cuda()
    output1 = model1(query, key, value)
    print(output1)

    model2 = nn.MultiheadAttention(d_model, h).cuda()
    output2, _ = model2(query, key, value)
    print(output2)

