import torch
import torch.nn as nn


class SelfAttention(nn.module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
