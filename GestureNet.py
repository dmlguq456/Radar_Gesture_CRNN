import torch.nn as nn
import torch.nn.functional as F
import torch

class GestureNet(nn.Module):
  def __init__(self):
    super(GestureNet, self).__init__()
    self.conv1 = nn.Conv2d(6, 16,kernel_size=(5,1))
    self.conv2 = nn.Conv2d(16,32,kernel_size=3,stride=2)
    self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
    self.batch_norm1 = nn.BatchNorm2d(16)
    self.batch_norm2 = nn.BatchNorm2d(32)
    self.fc1 = nn.Linear(64*4, 64)
    self.fc2 = nn.Linear(64, 7)
    self.rnn = nn.LSTM(64*4, 64*4, batch_first=True)


  def forward(self, x):
    x = self.conv1(x)
    x = self.batch_norm1(x)
    x = F.relu(x)
    x = self.conv2(x)
    x = self.batch_norm2(x)
    x = F.relu(x)
    x = self.conv3(x)
    # print(x.shape)
    x = x.permute(0,2,1,3)
    x = x.reshape(x.shape[0],x.shape[1],-1)
    x, (hn,cn) = self.rnn(x)
    x = self.fc1(x)
    x = F.relu(x)
    x = self.fc2(x) 
    # 5 X 72 X 7
    x = F.softmax(x, dim=2)
    output, idx = torch.max(x,dim=1)
    # 5 X 7
    return output
