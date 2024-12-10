import torch
import torch.nn as nn

class RNN2(nn.Module):
    
    def __init__(self, input_size=1, hidden_size=128, num_layers=4, num_classes=5, dropout=0.3):
        super(RNN2, self).__init__()

        # LSTM Layer
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=dropout
        )

        # Multi-Head Attention Layer
        self.attention_heads = nn.ModuleList([
            nn.Linear(hidden_size * 2, 1) for _ in range(4)  # 4 attention heads
        ])

        # Fully Connected Layer
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 8, 512),  # 4 heads * 2*hidden_size
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

        self.requires_reshape = True

    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)

        # Multi-Head Attention
        attention_outputs = [torch.softmax(head(lstm_out), dim=1) * lstm_out for head in self.attention_heads]
        context = torch.cat([torch.sum(out, dim=1) for out in attention_outputs], dim=1)

        # Fully Connected Layer
        out = self.fc(context)
        return out
