import torch
import torch.nn as nn
import torch.nn.functional as F

class StateEstimator(nn.Module):
    def __init__(self, num_obs=240, hidden_dimension=[256, 128, 64], output_size=6, dropout_prob=0.2):
        super(StateEstimator, self).__init__()
        self.fc1 = nn.Linear(num_obs, hidden_dimension[0])
        self.fc2 = nn.Linear(hidden_dimension[0], hidden_dimension[1])
        self.fc3 = nn.Linear(hidden_dimension[1], hidden_dimension[2])
        self.fc4 = nn.Linear(hidden_dimension[2], output_size)
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x
