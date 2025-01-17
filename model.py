import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.parameter import Parameter


class VIN(nn.Module):
    def __init__(self, config):
        super(VIN, self).__init__()
        self.config = config
        self.h = nn.Conv2d(
            in_channels=config.l_i,
            out_channels=config.l_h,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            bias=True)
        self.r = nn.Conv2d(
            in_channels=config.l_h,
            out_channels=1,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            bias=False)
        self.q = nn.Conv2d(
            in_channels=1,
            out_channels=config.l_q,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            bias=False)
        # self.fc = nn.Linear(in_features=config.l_q, out_features=1, bias=False)
        self.fc = nn.Linear(in_features=config.l_q, out_features=1, bias=False)
        self.w = Parameter(
            torch.zeros(config.l_q, 1, 3, 3), requires_grad=True)
        self.sm = nn.Softmax(dim=1)

    def forward(self, X, S1, S2, config):
        h = self.h(X)
        r = self.r(h)
        q = self.q(r)
        v, _ = torch.max(q, dim=1, keepdim=True)
        for i in range(0, config.k - 1):
            q = F.conv2d(
                torch.cat([r, v], 1),
                torch.cat([self.q.weight, self.w], 1),
                stride=1,
                padding=1)
            v, _ = torch.max(q, dim=1, keepdim=True)

        q = F.conv2d(
            torch.cat([r, v], 1),
            torch.cat([self.q.weight, self.w], 1),
            stride=1,
            padding=1)

        S1 = S1.squeeze(1)
        S2 = S2.squeeze(1)
        S1_grid = torch.zeros(S1.size())
        S2_grid = torch.zeros(S2.size())

        for i in range(S1.size(0)):
            for j in range(16):
                if ((-50+6.25*j)<S1[i]) and (S1[i]<-50+6.25*(j+1)):
                    S1_grid[i]=j

        for i in range(S2.size(0)):
            for j in range(16):
                if ((-50+6.25*j)<S2[i]) and (S2[i]<-50+6.25*(j+1)):
                    S2_grid[i]=j

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        S1_grid = S1_grid.to(device)
        S2_grid = S2_grid.to(device)

        slice_s1 = S1_grid.long().expand(config.imsize, 1, config.l_q, q.size(0))
        slice_s1 = slice_s1.permute(3, 2, 1, 0)
        q_out = q.gather(2, slice_s1).squeeze(2)

        slice_s2 = S2_grid.long().expand(1, config.l_q, q.size(0))#S2.long().expand(1, config.l_q, q.size(0))
        slice_s2 = slice_s2.permute(2, 1, 0)
        q_out = q_out.gather(2, slice_s2).squeeze(2)

        logits = self.fc(q_out)
        return logits, self.sm(logits)
