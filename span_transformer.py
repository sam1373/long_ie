from torch import nn


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
                nn.MultiheadAttention(hid_dim, num_heads)
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

        attns = []

        for i in range(self.num_layers):
            x1, attn = self.attn[i](x, context, context)

            attns.append(attn)

            x = x + x1
            x = self.norm[i](x)
            x = x + self.lin[i](x)

        return x, attns
