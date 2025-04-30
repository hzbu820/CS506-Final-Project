import torch.nn as nn
import torch

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])


#Double dropout layer version
# class LSTMModel(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
#         super().__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers

#         # LSTM 
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)

#         self.fc_hidden = nn.Linear(hidden_size, hidden_size // 2)
#         self.relu = nn.ReLU()
 
#         self.fc_out = nn.Linear(hidden_size // 2, output_size)

#     def forward(self, x):
#         h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
#         c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

#         out, _ = self.lstm(x, (h0, c0))
#         out = out[:, -1, :]  

#         out = self.fc_hidden(out)
#         out = self.relu(out)
#         out = self.fc_out(out)
#         return out
