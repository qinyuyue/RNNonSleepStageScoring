import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import metrics
import constant as C
from utils import evaluate
import totensor
from sleep_dataset import SleepDataset

if __name__ == '__main__':
	print("Test set loading")
	test_path = '../submission_test/testing.txt'
	filenames = totensor.read_set(test_path)
	print("Number of files to load: {}".format(len(filenames)))
	filepaths = ['../submission_test/' + f for f in filenames]
	data, labels = totensor.finish_tensor(filepaths)
	dataset = SleepDataset(data, labels)
	test_loader = DataLoader(dataset=dataset, batch_size=C.BATCH_SIZE, shuffle=False)

	best_model = torch.load("../submission_test/full_SleepRNN_6C_16U_3L.pth")
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(best_model.parameters())

	device = torch.device("cuda" if torch.cuda.is_available() and C.USE_CUDA else "cpu")
	best_model.to(device)
	criterion.to(device)

	loss, accuracy, results = evaluate(best_model, device, test_loader, criterion)
	y_true, y_pred = metrics.split_results(results)
	f1 = metrics.get_f1(y_true, y_pred)
	ck = metrics.get_kappa(y_true, y_pred)

	print("Loss")
	print("==============================================")
	print("Testing loss: {}".format(loss))

	print("Metrics")
	print("==============================================")
	print("Testing accuracy score:      {}".format(accuracy/100))
	print("Testing f1 score:            {}".format(f1))
	print("Testing Cohen's Kappa score: {}".format(ck))