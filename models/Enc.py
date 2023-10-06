import torch.nn as nn

class Enc(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Enc, self).__init__()
        self.fc_1 = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(p=0)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        x1 = self.fc_1(x)
        out = self.dropout(x1)
        return out