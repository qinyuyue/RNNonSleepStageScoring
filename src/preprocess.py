import os
import time
import multiprocessing as mp

import mne
import numpy as np
import pandas as pd
from lxml import etree
from numpy.fft import fft
from scipy.signal import detrend

import constant as C

"""
Reads raw data (EDF and XML files), does some data processing and feature
construction, then writes to csv files. Modify main to run in serial
or parallel.
"""


def get_signals(path):
	"""
	1) Reading edf file using MNE
	2) Band pass filtering
	3) Downsampling

	:param path: str, filepath to the EDF files (ending in '/')
	:return: dataframe of EEG, EEG2, and EMG signals
	"""
	mne.set_log_level('WARNING')

	raw_edf = mne.io.read_raw_edf(path, preload=True)
	channels = raw_edf.ch_names
	if 'EEG 2' in channels:
		raw_edf.rename_channels({'EEG 2': 'EEG2'})
	elif 'EEG(sec)' in channels:
		raw_edf.rename_channels({'EEG(sec)': 'EEG2'})
	elif 'EEG(SEC)' in channels:
		raw_edf.rename_channels({'EEG(SEC)': 'EEG2'})
	raw_edf.pick_channels(['EEG', 'EEG2', 'EMG'])         # Select channels
	raw_edf.filter(2, 30., fir_design='firwin')	          # Band filter
	raw_edf.resample(C.FINAL_SAMPLING_FREQ, npad='auto')  # Downsampling to 62 Hz

	return raw_edf.to_data_frame()


def get_labels(path):
	"""
	:param path: str, filepath to the XML files (ending in '/')
	:return: ndarray of labels for each second
	"""
	f = None
	try:
		f = open(path, 'rt')
		print("opened file")
	except FileNotFoundError:
		print('File at "{}" does not exist'.format(path))
	else:
		tree = etree.parse(f)
		root = tree.getroot()

		epoch_duration = int(root.find('EpochLength').text)  # Should be in seconds
		assert epoch_duration == 30, 'XML at "{}" does not have an epoch length of 30 seconds'.format(path)

		# Get all tags of EventType with 'Stages|Stages' as text, and then get their parents (ScoredEvent)
		stage_events = root.xpath('.//EventType[text()="Stages|Stages"]/..')

		# Initialize the label array
		last_start = int(float(stage_events[-1].find('Start').text))
		last_duration = int(float(stage_events[-1].find('Duration').text))
		length = (last_start + last_duration) * C.FINAL_SAMPLING_FREQ
		labels = np.full(length, -1)

		# Fill array
		i = 0
		for event in stage_events:
			raw_score = event.find('EventConcept').text   # Looks like this: 'Stage 1 sleep|1' or 'Wake|0'
			score = int(raw_score.split('|')[1])
			if score == 4:
				score = 3
			elif score == 5:
				score = 4
			duration = int(float(event.find('Duration').text))
			n = duration * C.FINAL_SAMPLING_FREQ
			labels[i:i+n] = np.full(n, score)
			i += n

		return labels

	finally:
		if f: f.close()   # Will close even after returning in else block


def add_both_labels(df, scores):
	"""
	Combines time series and labels into one df

	:param df: dataframe, output of get_signals() function
	:param scores: ndarray, output of get_labels() function
	:return: dataframe of columns: [seconds, EEG, EEG2, EMG, score]
	"""
	assert len(df) == len(scores)
	length = len(df)
	num_secs = length // C.FINAL_SAMPLING_FREQ

	df["score"] = scores

	seconds = np.arange(1, num_secs + 1)
	seconds = np.repeat(seconds, C.FINAL_SAMPLING_FREQ)
	df["seconds"] = seconds

	samples = np.arange(1, C.FINAL_SAMPLING_FREQ + 1)
	samples = np.tile(samples, num_secs)
	samples = samples.reshape(length)
	df["samples"] = samples

	df["seconds"] = df["seconds"].astype(str).str.cat(df["samples"].astype(str), sep=".")   # "second.sample" format
	df = df[["seconds", "EEG", "EEG2", "EMG", "score"]]

	# Some epochs are unscored (score=9). Removing those after assigning seconds to preserve the correct time
	df = df[df.score != 9]
	for s in df.score.unique():
		if s not in range(0, 5):
			raise ValueError("Unknown score of {} found".format(s))
	return df


def compute_fft(df):
	"""
	Computes the FFT of each epoch and then adds them as new columns

	:param df: dataframe, after signals and labels have been combined
	:return: dataframe with the FFT sequence added as new columns:
			[seconds, EEG, EEG2, EMG, EEG_f, EEG2_f, EMG_f, score]
	"""
	epoch_duration = 30

	new_cols = ['EEG_f', 'EEG2_f', 'EMG_f']
	df = df.assign(**dict.fromkeys(new_cols, np.nan))  # Adding 3 columns of NaN to the df

	n = epoch_duration * C.FINAL_SAMPLING_FREQ  # Number of rows per epoch
	for i in range(0, df.shape[0], n):
		np_seqs = df[['EEG', 'EEG2', 'EMG']].iloc[i:i+n].values  # Extracting time series data into ndarray
		np_seqsf = fft_allchannels(np_seqs)                      # Result is an ndarray of length approx n/2
		df.iloc[i:i + np_seqsf.shape[0], -3:] = np_seqsf         # Last 3 columns are EEG_f, EEG2_f, EMG_f

	df = df[['seconds', 'EEG', 'EEG2', 'EMG', 'EEG_f', 'EEG2_f', 'EMG_f', 'score']]  # Rearranging
	return df


def fft_allchannels(epoch):
	"""
	FFT the inputs
	:param sequences: ndarray, shape (sequence_length, n_features): time-series sequences
	:return: ndarray, shape (sequence_length, n_features): frequency-domain sequences
	"""
	assert epoch.ndim == 2
	sequence_length, n_features = epoch.shape
	epoch = detrend(epoch, axis=0, type='constant')	          # Subtract the mean to remove zero frequency content
	epoch_f = 1/sequence_length * np.abs(fft(epoch, axis=0))  # Real number FFT results in approx half size of data
	return epoch_f


def single_load(edf_file, xml_file):
	"""
	Loads a single pair of edf and xml files
	:param edf_file: str, filename of EDF file (excluding directory path)
	:param xml_file: str, filename of XML file (excluding directory path)
	:return: (process_id of the process running this load, time taken to run this load (seconds))
	"""
	start_time = time.time()
	edf_dir = C.RAW_EDF_DIR
	xml_dir = C.RAW_XML_DIR
	out_dir = C.PROCESSED_DIR

	file_id = edf_file.split('.')[0]
	print("Doing file {}".format(file_id))

	signals = get_signals(edf_dir + edf_file)
	labels = get_labels(xml_dir + xml_file)
	df = add_both_labels(signals, labels)
	df = compute_fft(df)
	df.to_csv(out_dir + "processed-" + file_id + ".csv", index=False)

	pid = os.getpid()
	time_taken = time.time() - start_time
	print("Done with file {} on process {} in time {}".format(file_id, pid, time_taken))

	return pid, int(time_taken)


def serial_load():
	"""
	:return: dataframe, time taken to run each file (seconds)
	"""
	edf_dir = C.RAW_EDF_DIR
	xml_dir = C.RAW_XML_DIR
	edf_filenames = [f for f in os.listdir(edf_dir) if os.path.isfile(edf_dir + f)]  # list of all files in dir
	xml_filenames = [f for f in os.listdir(xml_dir) if os.path.isfile(xml_dir + f)]

	times = np.zeros(len(edf_filenames))
	for i in range(0, len(edf_filenames)):
		pid, times[i] = single_load(edf_filenames[i], xml_filenames[i])
	df = pd.DataFrame(times, columns=[pid])
	return df


def parallel_load(n_processors=1):
	"""
	:param n_processors: int, number of cpu cores to run in parallel
	:return: dataframe, time taken to run each file by each process (seconds)
	"""
	# Collects all edf and xml filenames in the directories
	edf_dir = C.RAW_EDF_DIR
	xml_dir = C.RAW_XML_DIR
	edf_filenames = [f for f in os.listdir(edf_dir) if os.path.isfile(edf_dir + f)]  # list of all files in dir
	xml_filenames = [f for f in os.listdir(xml_dir) if os.path.isfile(xml_dir + f)]

	# Storing time taken by each process on each file
	results = {}
	def collect_time(r):
		pid = r[0]
		time_taken = r[1]
		if pid not in results:
			results[pid] = [time_taken]
		else:
			results[pid].append(time_taken)

	# Asynchronous multiprocessing
	pool = mp.Pool(n_processors)
	for i in range(0, len(edf_filenames)):
		pool.apply_async(single_load, args=(edf_filenames[i], xml_filenames[i]), callback=collect_time)
	pool.close()
	pool.join()

	# Compiling time metrics
	max_length = np.max([len(results[k]) for k in results])
	index = np.arange(0, max_length)
	res_df = pd.DataFrame(index=index, columns=results.keys())
	for k in results:
		times = np.array(results[k])
		buffer = np.full(max_length - len(times), np.nan)
		res_df[k] = np.concatenate((times, buffer))
	return res_df


if __name__ == '__main__':
	num_processors = 4
	print("Starting Parallel Loading with {} processors".format(num_processors))
	start = time.time()
	par_df = parallel_load(num_processors)
	par_time = int(time.time() - start)
	print("Total time taken: {}".format(par_time))
	vec = np.full(len(par_df), np.nan)
	vec[0] = par_time
	par_df['total'] = vec
	par_df.to_csv(C.OUTPUT_DIR + 'preprocess_par_timelogs.csv', index=False)

	# print("Starting Serial Loading")
	# start = time.time()
	# ser_df = serial_load()
	# ser_time = time.time() - start
	# print("Total time taken: {}".format(ser_time))
	# vec = np.full(len(ser_df), np.nan)
	# vec[0] = ser_time
	# ser_df['total'] = vec
	# ser_df.to_csv(C.OUTPUT_DIR + 'preprocess_ser_timelogs.csv', index=False)
