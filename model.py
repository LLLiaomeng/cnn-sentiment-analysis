import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TextCNN(nn.Module):
    def __init__(self, vec_dim, kernel_num, vec_num, label_num, kernel_list):
        super(TextCNN, self).__init__()
        chanel_num = 1
        self.convs = nn.ModuleList([nn.Sequential(
            nn.Conv2d(chanel_num, kernel_num, (kernel, vec_dim)),
            nn.ReLU(),
            nn.MaxPool2d((vec_num - kernel + 1, 1))
        )
            for kernel in kernel_list])
        self.fc = nn.Linear(kernel_num * len(kernel_list), label_num)
        self.dropout = nn.Dropout(0.5)
        self.sm = nn.Softmax(0)

    def forward(self, x):
        in_size = x.size(0)  
        out = [conv(x) for conv in self.convs]
        out = torch.cat(out, dim=1)
        out = out.view(in_size, -1)  
        out = self.dropout(out)
        out = self.fc(out)  
        return out
