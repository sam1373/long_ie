import torch
from torch import nn
import torch.nn.functional as F

from util import get_pairwise_idxs_separate

from util import RegLayer

import math

from attn_mod import MultiheadAttention



class ContextTransformer(nn.Module):

    def __init__(self, hid_dim, num_heads=4, num_layers=3, attn_scores_with_softmax=True):

        super().__init__()


        self.attn = []
        self.norm = []
        self.norm2 = []
        self.norm3 = []
        self.lin = []


        for i in range(num_layers):
            self.attn.append(
                MultiheadAttention(hid_dim, num_heads)
            )

            self.norm.append(
                nn.LayerNorm(hid_dim)
            )
            self.norm2.append(
                nn.LayerNorm(hid_dim)
            )
            self.norm3.append(
                nn.LayerNorm(hid_dim)
            )
            self.lin.append(
                nn.Linear(hid_dim, hid_dim)
            )

        self.attn = nn.ModuleList(self.attn)
        self.norm = nn.ModuleList(self.norm)
        self.norm2 = nn.ModuleList(self.norm2)
        self.norm3 = nn.ModuleList(self.norm3)
        self.lin = nn.ModuleList(self.lin)

        self.num_layers = num_layers

        self.attn_softmax = attn_scores_with_softmax

    def forward(self, context, x):

        #x[-2, :] += self.thr_emb
        #x[-1, :] += self.offload_emb
        #context[-2, :] += self.thr_emb
        #context[-1, :] += self.offload_emb

        attns = []

        for i in range(self.num_layers):

            x1, attn, attn_sm = self.attn[i](x, context, context)

            if self.attn_softmax:
                attns.append(attn_sm)
            else:
                attns.append(attn)
            #x1 = self.norm[i](x1)

            x = x + x1

            x = self.norm[i](x)

            x = x + self.lin[i](x)

            #x = self.norm2[i](x)

        return x, attns


