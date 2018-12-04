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
        self.fc1 = nn.Linear(in_features=9 * config.l_q+3, out_features=100, bias=False)
        self.fc2 = nn.Linear(in_features=100, out_features=50, bias=False)
        self.fc3 = nn.Linear(in_features=50, out_features=1, bias=False)
        self.w = Parameter(
            torch.zeros(config.l_q, 1, 3, 3), requires_grad=True)

    def forward(self, X, S1, S2, gamma, config):
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


        S1_grid = torch.zeros(S1.size())
        S2_grid = torch.zeros(S2.size())

        for i in range(S1.size(0)):
            for j in range(16):
                if ((0 + j) < S1[i]) and (S1[i] < 0 + (j + 1)): #if ((-50+6.25*j)<S1[i]) and (S1[i]<-50+6.25*(j+1)): #
                    S1_grid[i]=int(j)

        for i in range(S2.size(0)):
            for j in range(16):
                if ((16 - j) > S2[i]) and (S2[i] > 16 - (j + 1)):#if ((50-6.25*j)>S2[i]) and (S2[i]>50-6.25*(j+1)): #
                    S2_grid[i]=int(j)



        S1_grid = S1_grid.cuda()
        S2_grid = S2_grid.cuda()

        S11_grid = S1_grid - 1
        S13_grid = S1_grid + 1

        for i, data in enumerate(S11_grid):
            if data==-1:
                S11_grid[i]=0
        for i, data in enumerate(S13_grid):
            if data==16:
                S13_grid[i]=15

        S21_grid=S2_grid-1
        S23_grid=S2_grid+1

        for i, data in enumerate(S21_grid):
            if data==-1:
                S21_grid[i]=0
        for i, data in enumerate(S23_grid):
            if data==16:
                S23_grid[i]=15


        slice_s11 = S11_grid.long().expand(config.imsize, 1, config.l_q, q.size(0))
        slice_s12 = S1_grid.long().expand(config.imsize, 1, config.l_q, q.size(0))
        slice_s13 = S13_grid.long().expand(config.imsize, 1, config.l_q, q.size(0))

        slice_s11 = slice_s11.permute(3, 2, 0, 1)
        slice_s12 = slice_s12.permute(3, 2, 0, 1)
        slice_s13 = slice_s13.permute(3, 2, 0, 1)

        q_out1 = q.gather(3, slice_s11).squeeze(3)
        q_out2 = q.gather(3, slice_s12).squeeze(3)
        q_out3 = q.gather(3, slice_s13).squeeze(3)

        slice_s21 = S21_grid.long().expand(1, config.l_q, q.size(0))
        slice_s22 = (S2_grid).long().expand(1, config.l_q, q.size(0))
        slice_s23 = S23_grid.long().expand(1, config.l_q, q.size(0))

        slice_s21 = slice_s21.permute(2, 1, 0)
        slice_s22 = slice_s22.permute(2, 1, 0)
        slice_s23 = slice_s23.permute(2, 1, 0)

        q_out11 = q_out1.gather(2, slice_s21).squeeze(2)
        q_out12 = q_out1.gather(2, slice_s22).squeeze(2)
        q_out13 = q_out1.gather(2, slice_s23).squeeze(2)
        q_out21 = q_out2.gather(2, slice_s21).squeeze(2)
        q_out22 = q_out2.gather(2, slice_s22).squeeze(2)
        q_out23 = q_out2.gather(2, slice_s23).squeeze(2)
        q_out31 = q_out3.gather(2, slice_s21).squeeze(2)
        q_out32 = q_out3.gather(2, slice_s22).squeeze(2)
        q_out33 = q_out3.gather(2, slice_s23).squeeze(2)

        q_out = torch.cat([q_out11, q_out12, q_out13, q_out21, q_out22, q_out23, q_out31, q_out32, q_out33])
        q_out = torch.reshape(q_out, (9, S1.size(0), config.l_q))
        q_out = q_out.permute(1, 0, 2)
        q_out = q_out.reshape((S1.size(0), 9*config.l_q))
        v_out=torch.cat([q_out.permute(1,0),S1.view(-1,1).permute(1,0).float(),S2.view(-1,1).permute(1,0).float(),gamma.view(-1,1).permute(1,0).float()])
        v_out=v_out.permute(1,0)

        logits1 = F.relu(self.fc1(v_out))
        #logits = F.relu(self.fc1(q_out))
        #logits = F.dropout(logits, training=self.training)
        #logits2= self.fc2(logits1)
        logits2 = F.relu(self.fc2(logits1))
        #logits = self.fc3(logits2)
        logits = self.fc3(logits2)
        return logits
