import numpy as np
import constant as C


def read_set(file_path):
	"""
	:param file_path: full filepath to the TXT file that contains the data partition (train/test/valid) filenames
	:return: list of filenames (excluding directory path) for the data partition (train/test/valid)
	"""
	patients = []
	with open(file_path) as fp:
		for line in fp:
			line = line.split('\n')[0]
			patients.append(line)
	return patients


def make_tensor(file_path):
	"""
	:param file_path: full filepath to the CSV file
	:return: 3D array of shape (epoch, sequence, features)
	"""
	my_data  = np.genfromtxt(file_path, delimiter=',', skip_header=1)
	num_rows = my_data.shape[0]
	n = C.FINAL_SAMPLING_FREQ * C.S_PER_EPOCH  # Rows per epoch

	labels = my_data[:, -1]
	labels = np.reshape(labels, (num_rows // n, n, 1))  # Into shape (epoch, sequence, features)
	labels = labels[:, 0, 0]
	labels = labels.astype(np.int64)

	my_data = np.reshape(my_data[:, 1:-1], (num_rows // n, n, 6))  # Into shape (epoch, sequence, features)
	my_data = my_data.astype(np.float32)
	return my_data, labels


def finish_tensor(filepath_list):
	"""
	:param filepath_list: full filepath to the CSV files that should all be in one tensor
	:return: 3D array of shape (epoch, sequence, features)
	"""
	print("1/{} files: {}".format(len(filepath_list), filepath_list[0]))
	data_tensor, label_tensor = make_tensor(filepath_list[0])
	for i in range(1, len(filepath_list)):
		print("{}/{} files: {}".format(i, len(filepath_list), filepath_list[i]))
		data_temp, label_temp = make_tensor(filepath_list[i])
		data_tensor = np.concatenate((data_tensor, data_temp))
		label_tensor = np.concatenate((label_tensor, label_temp))
	print("Final tensor shape: {}".format(data_tensor.shape))
	return data_tensor, label_tensor