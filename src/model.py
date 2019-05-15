import torch.nn as nn
import torch.nn.functional as F


class SleepRNN6(nn.Module):
	def __init__(self, hidden_units=16, num_layers=2, dropout=0.3):
		super(SleepRNN6, self).__init__()
		self.rnn = nn.GRU(input_size=6,
		                  hidden_size=hidden_units,
		                  num_layers=num_layers,
		                  batch_first=True,
		                  dropout=dropout)
		self.fc = nn.Linear(in_features=hidden_units, out_features=5)
		self.details = "6C_{}U_{}L".format(hidden_units, num_layers)  # Num channels, num hidden units, num layers

	def forward(self, x):
		x, _ = self.rnn(x)
		x = self.fc(x[:, -1, :])
		return x


class SleepRNN3(nn.Module):
	def __init__(self, hidden_units=16, num_layers=2, dropout=0.3):
		super(SleepRNN3, self).__init__()
		self.rnn = nn.GRU(input_size=6,
		                  hidden_size=hidden_units,
		                  num_layers=num_layers,
		                  batch_first=True,
		                  dropout=dropout)
		self.fc = nn.Linear(in_features=hidden_units, out_features=5)
		self.details = "3C_{}U_{}L".format(hidden_units, num_layers)  # Num channels, num hidden units, num layers


	def forward(self, x):
		x, _ = self.rnn(x)
		x = self.fc(x[:, -1, :])
		return x