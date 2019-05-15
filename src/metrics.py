import numpy as np
import pandas as pd

import constant as C
import matplotlib.pyplot as plt
from  sklearn.metrics import f1_score, cohen_kappa_score
import seaborn as sns


def split_results(results):
	splitted = list(zip(*results))
	# print(splitted.shape)
	# set
	y_true = list(splitted[0])
	y_pred = list(splitted[1])
	# array
	y_true = np.asarray(y_true)
	y_pred = np.asarray(y_pred)

	return y_true, y_pred


def get_f1(y_true, y_pred):
	return f1_score(y_true, y_pred, average='micro')

def get_kappa(y_true, y_pred):
	return cohen_kappa_score(y_true, y_pred)

if __name__ == '__main__':
	results = pd.read_csv('../cross-validation/output/0results_6C_16U_3L.csv')
	y_true = results['true']
	y_pred = results['pred']
	f1 = get_f1(y_true, y_pred)
	cohen_kappa = get_kappa(y_true, y_pred)
	print("=====")
	print(f1)
	print("=====")
	print(cohen_kappa)

	CV_DIR = '../cross-validation/output/'
	unit_range = [8, 16, 24, 32, 64]
	layer_range = [2, 3, 4]
	f1 = np.zeros((len(unit_range), len(layer_range)))
	ck = np.zeros((len(unit_range), len(layer_range)))
	for i in range(0, len(unit_range)):
		for j in range(0, len(layer_range)):
			filename = '0results_6C_{}U_{}L.csv'.format(unit_range[i], layer_range[j])
			results = pd.read_csv(CV_DIR + filename)
			y_true = results['true']
			y_pred = results['pred']
			f1[i, j] = 100 * get_f1(y_true, y_pred)
			ck[i, j] = 100 * get_kappa(y_true, y_pred)

	"""
	Plotting
	"""
	xticks = ['{} Layers'.format(l) for l in layer_range]
	yticks = ['{} Units'.format(u) for u in unit_range]

	# F1
	plt.figure()
	ax = sns.heatmap(f1,
	                 annot=True, fmt='.2f',
	                 xticklabels=xticks, yticklabels=yticks,
	                 linewidths=.5, cmap="YlGnBu")
	ax.set_title('Validation F1 for Different Combinations of Hyperparameters')
	plt.tight_layout()
	plt.savefig(C.GRAPHS_DIR + 'hyperparameter_tuning_f1.png', dpi=300)

	# Cohen's Kappa
	plt.figure()
	ax = sns.heatmap(ck,
	                 annot=True, fmt='.2f',
	                 xticklabels=xticks, yticklabels=yticks,
	                 linewidths=.5, cmap="YlGnBu")
	ax.set_title('Validation Cohen\'s Kappa for Different Combinations of Hyperparameters')
	plt.tight_layout()
	plt.savefig(C.GRAPHS_DIR + 'hyperparameter_tuning_ck.png', dpi=300)
	plt.show()
