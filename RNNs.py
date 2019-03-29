import torch
from torch.autograd import Variable
import torch.nn as nn



class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, self.hidden_size, self.num_layers, batch_first=True, dropout=0.2)
        #self.gru = nn.GRU(input_size, self.hidden_size, self.num_layers, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(hidden_size, int(hidden_size/2))
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(int(hidden_size/2), int(hidden_size/2))
        self.fc3 = nn.Linear(int(hidden_size/2), num_classes)
    
    def forward(self, x):
        x = x.float()
        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).float()).cuda() 
        c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).float()).cuda()
        out, _ = self.lstm(x, (h0,c0)) 
        out = self.relu(self.fc1(out[:, -1, :]))
        out = self.relu(self.fc2(out))
        out = self.fc3(out) 
        return out
