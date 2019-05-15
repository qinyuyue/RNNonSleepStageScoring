import os
import numpy as np
import constant as C


def sample(file_list, amount=20):
	"""
	Returns an ndarray of samples, and an ndarray of the remainder
	"""
	np.random.shuffle(file_list)
	return file_list[:amount], file_list[amount:]


def split(file_list, test_frac=0.2):
	np.random.shuffle(file_list)
	cutoff = int(test_frac * len(file_list))
	test_list  = file_list[:cutoff]
	train_list = file_list[cutoff:]
	return test_list, train_list


def write_list(lists, file_name):
	with open(file_name, 'w') as f:
		for item in lists:
			f.write("%s\n" % item)


if __name__ == '__main__':
	processed_dir = C.PROCESSED_DIR

	csv_filenames = [f for f in os.listdir(processed_dir) if os.path.isfile(processed_dir + f)]  # list of all files in dir
	csv_filenames = np.array(csv_filenames)

	# Partitions for final model evaluation
	training_path = C.SPLITS_DIR + "training.txt"
	testing_path = C.SPLITS_DIR + "testing.txt"

	training_path   = C.SPLITS_DIR + "training.txt"
	testing_path    = C.SPLITS_DIR + "testing.txt"

	test_list, train_list = split(csv_filenames, 0.2)

	write_list(train_list, training_path)
	write_list(test_list, testing_path)

	# Partitions for cross-validation
	cv_test_list, cv_tv_list = sample(csv_filenames, 5)
	write_list(cv_test_list, C.SPLITS_DIR + 'cv_test.txt')
	for i in range(0, 3):
		cv_train_list, cv_rem_list = sample(cv_tv_list, 20)
		cv_valid_list, _           = sample(cv_rem_list, 5)
		write_list(cv_train_list, C.SPLITS_DIR + 'cv_train{}.txt'.format(i))
		write_list(cv_valid_list, C.SPLITS_DIR + 'cv_valid{}.txt'.format(i))