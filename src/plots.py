import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

import constant as C
from preprocess import get_labels


def plot_parallel_processing(save=False):
	"""
	Plotting time taken for processing data in python serial, vs python parallel vs spark
	"""
	# Reading data
	par = pd.read_csv(C.OUTPUT_DIR + 'preprocess_par_timelogs.csv')
	total_par = par['total'][0]/3600  # HOURS
	par = par.drop(columns=['total'])
	n_processes = par.shape[1]
	par_process_time = [par.iloc[:, i].dropna().values/60 for i in range(0, n_processes)]  #MINUTES

	ser = pd.read_csv(C.OUTPUT_DIR + 'preprocess_ser_timelogs.csv')
	total_ser = ser['total'][0]/3600  # HOURS
	ser = ser.drop(columns=['total'])
	ser_process_time = ser.iloc[:, 0].values/60  # MINUTES

	spk = pd.read_csv(C.OUTPUT_DIR + 'preprocess_spk_timelogs.csv')
	total_spk = 884/60  # HOURS (hard coded hours for 231 patients since we ran into errors on 232th patient)
	spk_process_time = spk['time'].values/60  # MINUTES

	# (Parallel) Time taken per process
	plt.figure()
	for i in range(0, n_processes):
		plt.plot(par_process_time[i], label='Process #{}'.format(i+1))
	plt.legend()
	plt.title("(Parallel) Time Taken per Process")
	plt.xlabel("File Processed")
	plt.ylabel("Time Taken(min)")
	if save:
		plt.savefig(C.GRAPHS_DIR + 'preprocess_1.png', dpi=300)

	# (Parallel + Series + Spark) Average time taken per process
	tic_loc = np.arange(2 + n_processes)
	avg_time = [np.mean(spk_process_time), np.mean(ser_process_time)]
	labels = ['Spark', 'Serial\n(Python)']
	color = ['#79E070', '#E88D67']
	for i in range(0, n_processes):
		avg_time.append(np.mean(par_process_time[i]))
		labels.append('Parallel #{}\n(Python)'.format(i+1))
		color.append('#7B8CDE')
	plt.figure()
	plt.bar(tic_loc, avg_time, tick_label=labels, color=color, align='center')
	plt.title("Average Time Taken On Each File Per Process")
	plt.ylabel("Time Taken(min)")
	if save:
		plt.savefig(C.GRAPHS_DIR + 'preprocess_2.png', dpi=300)

	# (Parallel + Series + Spark)
	plt.figure()
	plt.bar(x=[0, 1, 2], height=[total_spk, total_ser, total_par], tick_label=['Spark(est.)', 'Serial', 'Parallel'],
	        color=['#79E070', '#E88D67', '#7B8CDE'], align='center')
	plt.title("Total Time Taken For Dataset")
	plt.ylabel("Time Taken(h)")
	if save:
		plt.savefig(C.GRAPHS_DIR + 'preprocess_3.png', dpi=300)


def plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies, fold="", model_details=None):
	"""
	Plotting accuracy and loss curves

	:param fold: cross-validation fold.
	:param model_details: string of model parameters
	"""
	if model_details:
		ch, units, layers = model_details.split('_')

	# Loss
	plt.figure()
	plt.plot(np.arange(len(train_losses)), train_losses, label='Train')
	plt.plot(np.arange(len(valid_losses)), valid_losses, label='Validation')
	plt.ylabel('Loss')
	plt.xlabel('epoch')
	plt.legend(loc="best")
	if model_details:
		plt.title('Loss Curves for {} ch, {} units, {} layers'.format(ch, units, layers))
		plt.savefig(C.GRAPHS_DIR + '{}loss_curves_{}.png'.format(fold, model_details))
	else:
		plt.title('Loss Curves')
		plt.savefig(C.GRAPHS_DIR + 'loss_curves.png', dpi=300)

	# Accuracy
	plt.figure()
	plt.plot(np.arange(len(train_accuracies)), train_accuracies, label='Train')
	plt.plot(np.arange(len(valid_accuracies)), valid_accuracies, label='Validation')
	plt.ylabel('Accuracy')
	plt.xlabel('epoch')
	plt.legend(loc="best")
	if model_details:
		plt.title('Accuracy Curves for {} ch, {} units, {} layers'.format(ch, units, layers))
		plt.savefig(C.GRAPHS_DIR + '{}accu_curves_{}.png'.format(fold, model_details))
	else:
		plt.title('Accuracy Curves')
		plt.savefig(C.GRAPHS_DIR + 'accu_curves.png', dpi=300)


def plot_confusion_matrix(results, class_names, fold="", model_details=None):
	"""
	Plotting the confusion matrix

	:param fold: cross-validation fold.
	:param model_details: string of model parameters
	:return:
	"""
	if model_details:
		ch, units, layers = model_details.split('_')

	true_y, pred_y = zip(*results)  # Unzip
	true_y = list(true_y)
	pred_y = list(pred_y)
	cm = confusion_matrix(true_y, pred_y)

	## Plotting code adapted from:
	# https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
	cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalizing matrix values

	fig, ax = plt.subplots()
	im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
	ax.figure.colorbar(im, ax=ax)
	# We want to show all ticks...
	ax.set(xticks=np.arange(cm.shape[1]),
	       yticks=np.arange(cm.shape[0]),
	       # ... and label them with the respective list entries
	       xticklabels=class_names, yticklabels=class_names,
	       title='Confusion Matrix' if model_details == None else 'Confusion Matrix for {} ch, {} units, {} layers'.format(ch, units, layers),
	       ylabel='True label',
	       xlabel='Predicted label')

	# Rotate the tick labels and set their alignment.
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
	         rotation_mode="anchor")

	# Loop over data dimensions and create text annotations.
	fmt = '.2f'
	thresh = cm.max() / 2.
	for i in range(cm.shape[0]):
		for j in range(cm.shape[1]):
			ax.text(j, i, format(cm[i, j], fmt),
			        ha="center", va="center",
			        color="white" if cm[i, j] > thresh else "black")
	fig.tight_layout()

	if model_details:
		plt.savefig(C.GRAPHS_DIR + '{}confusion_matrix_{}.png'.format(fold, model_details), dpi=300)
	else:
		plt.savefig(C.GRAPHS_DIR + 'confusion_matrix.png', dpi=300)


def plot_data_distribution(fraction=0.1):
	"""
	Plotting distribution of sleep stages in data

	:param fraction: fraction of data to sample for the plot
	"""
	# Sampling from full dataset
	assert 0 < fraction <= 1
	all_xml = [f for f in os.listdir(C.RAW_XML_DIR) if os.path.isfile(C.RAW_XML_DIR + f)]
	all_xml = np.array(all_xml)
	np.random.shuffle(all_xml)
	cutoff = int(fraction * len(all_xml))
	samples = all_xml[:cutoff]

	# Getting labels for all epochs from each patient
	n = C.FINAL_SAMPLING_FREQ * 30  # Rows per epoch
	all_labels = []
	for path in samples:
		labels = get_labels(C.RAW_XML_DIR + path)
		num_rows = len(labels)

		labels = np.reshape(labels, (num_rows // n, n, 1))  # Into shape (epoch, sequence, features)
		labels = labels[:, 0, 0]
		labels = labels.astype(np.int64)

		all_labels.extend(labels)

	# Computing
	scores, counts = np.unique(np.array(all_labels), return_counts=True)
	percentage = []

	print("Number of patients: {}".format(len(samples)))
	print("Number of epochs: {}:".format(len(all_labels)))
	print("Score, Counts, Percentage")
	for i in range(0, len(scores)):
		print("{}, {}, {}".format(scores[i], counts[i], counts[i] / len(all_labels) * 100))
		percentage.append(counts[i] / len(all_labels) * 100)

	# Plotting
	x_pos = np.arange(len(scores))
	x_labels = ['Wake', 'N1', 'N2', 'N3', 'REM']

	plt.figure()
	plt.bar(x_pos, percentage, align='center', alpha=0.5)
	plt.xticks(x_pos, x_labels)
	plt.title('Distribution of Sleep Stages')
	plt.ylabel('Percentage')
	plt.savefig(C.GRAPHS_DIR + "stage_distribution.png")


def plot_tuning(accuracy_matrix, unit_range, layer_range):
	"""
	Plot a heatmap

	:param accuracy_matrix: ndarray of tuning accuracies to plot. shape: (num unit par, num layer par)
	:param unit_range: list of hidden unit params
	:param layer_range list of layer params
	"""
	xticks = ['{} Layers'.format(l) for l in layer_range]
	yticks = ['{} Units'.format(u) for u in unit_range]
	ax = sns.heatmap(accuracy_matrix,
	                 annot=True, fmt='.2f',
					 xticklabels=xticks, yticklabels=yticks,
					 linewidths=.5, cmap="YlGnBu")
	ax.set_title('Validation Accuracy for Different Combinations of Hyperparameters')
	plt.tight_layout()
	plt.savefig(C.GRAPHS_DIR + 'hyperparameter_tuning.png', dpi=300)
