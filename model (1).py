import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class LSTMNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout, num_layers=1):
        super(LSTMNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstms = nn.ModuleList([nn.LSTM(input_dim if i == 0 else hidden_dim, hidden_dim, batch_first=True) for i in range(num_layers)])
        #self.batch_norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)

        self.apply(self.xavier_init)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        out = x
        for lstm in self.lstms:
            out, (hn, cn) = lstm(out, (h0, c0))
            out = self.dropout(out)
        out = self.fc2(F.leaky_relu(self.fc1(out[:, -1, :])))
        return out
    
    def xavier_init(self, m):
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.conv_layers = self._create_conv_layers(input_size, num_channels, kernel_size, dropout)
        self.fc = nn.Linear(num_channels[-1], output_size)

    def _create_conv_layers(self, input_size, num_channels, kernel_size, dropout):
        layers = []
        for i in range(len(num_channels)):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i - 1]
            layers += [
                CausalDilatedConv1d(in_channels, num_channels[i], kernel_size, dilation=dilation_size),
                nn.Tanh(),
                nn.Dropout(dropout)
            ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_layers(x)
        x = F.adaptive_avg_pool1d(x, 1).squeeze(2)
        x = self.fc(x)
        return x

class CausalDilatedConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(CausalDilatedConv1d, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation)

    def forward(self, x):
        x = self.conv(x)
        return x[:, :, :-(self.padding * x.size(2) // x.size(2))]

    
# class LSTMNet(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.2):
#         super(LSTMNet, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
#         self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
#         self.fc2 = nn.Linear(hidden_dim // 2, output_dim)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         batch_size = x.size(0)
#         h0 = torch.zeros(1, batch_size, self.hidden_dim).to(x.device)
#         c0 = torch.zeros(1, batch_size, self.hidden_dim).to(x.device)
#         out, _ = self.lstm(x, (h0, c0))
#         out = self.dropout(torch.relu(self.fc1(out[:, -1, :])))
#         out = self.fc2(out)
#         return out
# class LSTMNet(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, dropout_prob):
#         super(LSTMNet, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
#         self.fc1 = nn.Linear(hidden_dim*2, hidden_dim//2)
#         self.dropout = nn.Dropout(dropout_prob)
#         self.fc2 = nn.Linear(hidden_dim//2, output_dim)

#         self.apply(self.xavier_init)

#     def forward(self, x):
#         batch_size = x.size(0)
#         h0 = torch.zeros(2, batch_size, self.hidden_dim).to(x.device)
#         c0 = torch.zeros(2, batch_size, self.hidden_dim).to(x.device)
#         out, _ = self.lstm(x, (h0, c0))
#         out = self.dropout(out)
#         out = self.fc2(torch.relu(self.fc1(out[:, -1, :])))
#         return out
    
#     def xavier_init(self, m):
#         if isinstance(m, nn.Linear):
#             init.xavier_uniform_(m.weight)
#             if m.bias is not None:
#                 init.constant_(m.bias, 0)

# class MLP(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super(MLP, self).__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(hidden_dim, output_dim)

#     def forward(self, x):
#         x = x.reshape(x.shape[0], -1)
#         out = self.fc1(x)
#         out = self.relu(out)
#         out = self.fc2(out)
#         return out


