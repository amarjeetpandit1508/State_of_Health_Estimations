import torch
import torch.nn as nn


class RecurrentNet(nn.Module):
    """
    Generic RNN model for sequence-to-one prediction using GRU or LSTM.
    """
    def __init__(self, input_dim, hidden_dim, output_dim,
                 n_layers=1, dropout=0.2, rnn_type='GRU'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.rnn_type = rnn_type.upper()

        # Choose between GRU and LSTM dynamically
        if self.rnn_type == 'GRU':
            self.rnn = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=n_layers,
                batch_first=True,
                dropout=dropout if n_layers > 1 else 0
            )
        elif self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=n_layers,
                batch_first=True,
                dropout=dropout if n_layers > 1 else 0
            )
        else:
            raise ValueError("rnn_type must be either 'GRU' or 'LSTM'.")

        # Output layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, input, hidden):
        """
        Forward pass through the network.
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            hidden: Hidden (and cell) state(s)
        Returns:
            output: Predicted value(s)
            hidden: Updated hidden (and cell) state(s)
        """
        out, hidden = self.rnn(input, hidden)
        # Take the output from the last time step
        out = self.relu(out[:, -1, :])
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size):
        """
        Initialize hidden states to zeros.
        """
        weight = next(self.parameters()).data

        if self.rnn_type == 'LSTM':
            hidden = (
                weight.new_zeros(self.n_layers, batch_size, self.hidden_dim),
                weight.new_zeros(self.n_layers, batch_size, self.hidden_dim)
            )
        else:  # GRU
            hidden = weight.new_zeros(self.n_layers, batch_size, self.hidden_dim)

        return hidden