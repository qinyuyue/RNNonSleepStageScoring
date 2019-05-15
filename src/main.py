import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import totensor
import constant as C
from model import SleepRNN3, SleepRNN6
from sleep_dataset import SleepDataset
from utils import train, evaluate
from plots import plot_learning_curves, plot_confusion_matrix


def import_to_dataloader(split_path):
	"""

	:param set_path: str, path to txt file that contains the filenames of the data to be imported into the DataLoader
	:param dir: str, directory where the filenames of the data are
	:return: torch.utils.data.DataLoader
	"""
	filenames = totensor.read_set(split_path)
	print("Number of files to load: {}".format(len(filenames)))
	filepaths = [C.PROCESSED_DIR + f for f in filenames]
	data, labels = totensor.finish_tensor(filepaths)
	dataset = SleepDataset(data, labels)
	loader = DataLoader(dataset=dataset, batch_size=C.BATCH_SIZE, shuffle=False)
	return loader


if __name__ == '__main__':
	if C.USE_CUDA and torch.cuda.is_available():
		print("Using CUDA")
		print("Device ID:   {}".format(torch.cuda.current_device()))
		print("Device name: {}".format(torch.cuda.get_device_name(0)))
	else:
		print("Using CPU")

	torch.manual_seed(0)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(0)

	"""
	SINGLE MODEL TRAINING (COMMENT OUT IF NOT USING)
	"""
	# Data loading
	print("Train set loading")
	train_path = C.SPLITS_DIR + 'training.txt'
	train_loader = import_to_dataloader(train_path)

	print("Valid set loading")
	valid_path = C.SPLITS_DIR + 'validation.txt'
	valid_loader = import_to_dataloader(valid_path)

	# print("Test set loading")
	# test_path = C.SPLITS_DIR + 'testing.txt'
	# test_loader = import_to_dataloader(test_path)

	train_start = time.time()
	model = SleepRNN6(hidden_units=16, num_layers=3)
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters())

	device = torch.device("cuda" if torch.cuda.is_available() and C.USE_CUDA else "cpu")
	model.to(device)
	criterion.to(device)

	# Model training and validation
	best_val_acc = 0.0
	train_losses, train_accuracies = [], []
	valid_losses, valid_accuracies = [], []
	with open(C.OUTPUT_DIR + '{}training_metrics_{}.csv'.format("full_", model.details), 'w') as f:
		# Open a new metrics file to write to after each epoch of training
		f.write("time,train_loss,valid_loss,train_acc,valid_acc\n")
		f.close()

	for epoch in range(C.NUM_EPOCHS):
		plt.close("all")
		start = time.time()
		train_loss, train_accuracy = train(model, device, train_loader, criterion, optimizer, epoch)
		valid_loss, valid_accuracy, valid_results = evaluate(model, device, valid_loader, criterion)
		time_taken = time.time() - start

		train_losses.append(train_loss)
		valid_losses.append(valid_loss)

		train_accuracies.append(train_accuracy)
		valid_accuracies.append(valid_accuracy)

		with open(C.OUTPUT_DIR + '{}training_metrics_{}.csv'.format("full_", model.details), 'a') as f:
			f.write("{},{},{},{},{}\n".format(time_taken, train_loss, valid_loss, train_accuracy,
			                                  valid_accuracy))
			f.close()

		is_best = valid_accuracy > best_val_acc  # let's keep the model that has the best accuracy, but you can also use another metric.
		if is_best:
			best_val_acc = valid_accuracy
			torch.save(model, os.path.join(C.OUTPUT_DIR, "{}SleepRNN_{}.pth".format("full_", model.details)))

			# Save results and confusion matrix in case of time-out
			with open(C.OUTPUT_DIR + "{}results_{}.csv".format("full_", model.details), "w") as f:
				f.write("true,pred\n")
				for r in valid_results:
					f.write("{},{}\n".format(r[0], r[1]))
				f.close()

			class_names = ['0', '1', '2', '3', '4']
			plot_confusion_matrix(valid_results, class_names, "full_", model.details)

	# plot learning curves
	plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies, "full_", model.details)


	"""
	HYPERPARAMETER TUNING (COMMENT OUT IF NOT USING)
	"""
	# unit_range = [8, 16, 24, 32, 64]
	# layer_range = [2, 3, 4]
	# for fold in range(0, 3):
	# 	print("FOLD #{}".format(fold))
	# 	# Data loading
	# 	print("Train set loading")
	# 	train_path = C.SPLITS_DIR + 'cv_train{}.txt'.format(fold)
	# 	train_loader = import_to_dataloader(train_path)
	#
	# 	print("Valid set loading")
	# 	valid_path = C.SPLITS_DIR + 'cv_valid{}.txt'.format(fold)
	# 	valid_loader = import_to_dataloader(valid_path)
	#
	# 	# Model setup
	# 	for u in unit_range:
	# 		for l in layer_range:
	# 			train_start = time.time()
	# 			model = SleepRNN6(hidden_units=u, num_layers=l)
	# 			criterion = nn.CrossEntropyLoss()
	# 			optimizer = optim.Adam(model.parameters())
	#
	# 			device = torch.device("cuda" if torch.cuda.is_available() and C.USE_CUDA else "cpu")
	# 			model.to(device)
	# 			criterion.to(device)
	#
	# 			# Model training and validation
	# 			best_val_acc = 0.0
	# 			train_losses, train_accuracies = [], []
	# 			valid_losses, valid_accuracies = [], []
	# 			with open(C.OUTPUT_DIR + '{}training_metrics_{}.csv'.format(fold, model.details), 'w') as f:
	# 				# Open a new metrics file to write to after each epoch of training
	# 				f.write("time,train_loss,valid_loss,train_acc,valid_acc\n")
	# 				f.close()
	#
	# 			for epoch in range(C.NUM_EPOCHS):
	# 				plt.close("all")
	# 				start = time.time()
	# 				train_loss, train_accuracy = train(model, device, train_loader, criterion, optimizer, epoch)
	# 				valid_loss, valid_accuracy, valid_results = evaluate(model, device, valid_loader, criterion)
	# 				time_taken = time.time() - start
	#
	# 				train_losses.append(train_loss)
	# 				valid_losses.append(valid_loss)
	#
	# 				train_accuracies.append(train_accuracy)
	# 				valid_accuracies.append(valid_accuracy)
	#
	# 				with open(C.OUTPUT_DIR + '{}training_metrics_{}.csv'.format(fold, model.details), 'a') as f:
	# 					f.write("{},{},{},{},{}\n".format(time_taken, train_loss, valid_loss, train_accuracy,
	# 					                                  valid_accuracy))
	# 					f.close()
	#
	# 				is_best = valid_accuracy > best_val_acc  # let's keep the model that has the best accuracy, but you can also use another metric.
	# 				if is_best:
	# 					best_val_acc = valid_accuracy
	# 					torch.save(model, os.path.join(C.OUTPUT_DIR, "{}SleepRNN_{}.pth".format(fold, model.details)))
	#
	# 					# Save results and confusion matrix in case of time-out
	# 					with open(C.OUTPUT_DIR + "{}results_{}.csv".format(fold, model.details), "w") as f:
	# 						f.write("true,pred\n")
	# 						for r in valid_results:
	# 							f.write("{},{}\n".format(r[0], r[1]))
	# 						f.close()
	#
	# 					class_names = ['0', '1', '2', '3', '4']
	# 					plot_confusion_matrix(valid_results, class_names, fold, model.details)
	#
	# 			# plot learning curves
	# 			plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies, fold, model.details)