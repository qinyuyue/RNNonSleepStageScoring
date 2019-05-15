from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.sql.functions import monotonically_increasing_id, split as f_split, pandas_udf, PandasUDFType
from pyspark.sql.types import *

import os
import time
import shutil
import multiprocessing as mp

import mne
import numpy as np
import pandas as pd
from lxml import etree
from numpy.fft import fft
from scipy.signal import detrend
# from scipy.fftpack import fft
# print("hello world")

# configuration
conf = SparkConf()
conf.setMaster('spark://10.0.0.4:7077')
conf.setAppName('spark-basic')
conf.set("spark.sql.shuffle.partitions", 18)
sc = SparkContext(conf=conf)

sqlcont = SQLContext(sc)

final_sampling_freq = 62
edf_dir = "../data/5percent/edfs/shhs1/"
xml_dir = "../data/5percent/annotations/shhs1/"
# edf_dir = "../data/edf/"
# xml_dir = "../data/xml/"


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
	raw_edf.resample(final_sampling_freq, npad='auto')  # Downsampling to 62 Hz
	pd_df = raw_edf.to_data_frame()
	sdf = sqlcont.createDataFrame(pd_df)
	return sdf

def get_labels(path):
	"""
	Get labels(score) from XML file
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
		length = (last_start + last_duration) * final_sampling_freq
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
			n = duration * final_sampling_freq
			labels[i:i+n] = np.full(n, score)
			i += n

		return labels

	finally:
		if f: f.close()   # Will close even after returning in else block

def add_both_labels(df, labels):
	"""
	Combines time series and labels into one df
	:param df: dataframe, output of get_signals() function
	:param scores: ndarray, output of get_labels() function
	:return: dataframe of columns: [seconds, EEG, EEG2, EMG, score]
	"""

	length = len(labels)
	num_secs = length // final_sampling_freq
	# numpy array of seconds
	seconds = np.arange(1, num_secs + 1)
	seconds = np.repeat(seconds, final_sampling_freq)
	seconds = seconds.astype(str)
	# numpy array of epochs
	epochs = np.arange(1, num_secs//30)
	epochs = np.repeat(epochs, 30 * final_sampling_freq)
	# numpy array of dot
	dots = np.array(['.' for _ in range(length)])
	# numpy array of samples
	samples = np.arange(1, final_sampling_freq + 1)
	samples = np.tile(samples, num_secs)
	samples = samples.reshape(length)
	samples = samples.astype(str)

	# concatenate each line of seconds, dots and samples
	combined = np.core.defchararray.add(seconds,dots)
	combined = np.core.defchararray.add(combined,samples)

	# df of seconds and labels
	seconds = sqlcont.createDataFrame(pd.DataFrame(combined),schema=['seconds'])
	epochs = sqlcont.createDataFrame(pd.DataFrame(epochs),schema=['epochs'])
	labels = sqlcont.createDataFrame([(int(item),) for item in labels], ['score'])

	# join 3 dataframes together
	df = df.withColumn("row_idx", monotonically_increasing_id())
	seconds = seconds.withColumn("row_idx", monotonically_increasing_id())
	epochs = epochs.withColumn("row_idx", monotonically_increasing_id())
	seconds = seconds.join(epochs, seconds.row_idx == epochs.row_idx).drop(seconds[-1])
	df = seconds.join(df, seconds.row_idx == df.row_idx).drop(df[-1])
	labels = labels.withColumn("row_idx", monotonically_increasing_id())
	df = df.join(labels, df.row_idx == labels.row_idx).drop("row_idx")

	# df = df[df.score != 9]
	df = df.filter(df.score != 9)
	df = df.repartition(18, "epochs")
	return df

def serial_load():
	"""
	:return: dataframe, time taken to run each file (seconds)
	"""
	times = np.zeros(len(edf_filenames))
	for i in range(0, len(edf_filenames)):
		pid, times[i] = single_load(edf_filenames[i], xml_filenames[i])
		with open('../output/spark_time_metrics_all.csv', 'a+') as f:
			# Open a new metrics file to write to after each epoch of training
			f.write("%s, %s\n" %(pid, times[i]))
			f.close()


def single_load(edf_file, xml_file):
	"""
	Loads a single pair of edf and xml files
	:param edf_file: str, filename of EDF file (excluding directory path)
	:param xml_file: str, filename of XML file (excluding directory path)
	:return: (process_id of the process running this load, time taken to run this load (seconds))
	"""
	start_time = time.time()
	processed_dir = "../data/processed/"

	file_id = edf_file.split('.')[0]
	print("Doing file {}".format(file_id))
	signals = get_signals(edf_dir + edf_file)
	labels = get_labels(xml_dir + xml_file)
	df = add_both_labels(signals, labels)
	outputs = df.groupby("epochs").apply(sci_fft)
	# outputs.toPandas().to_csv("../data/processed/processed_" + file_id + ".csv")
	output_path = "../data/processed/processed_" + file_id
	outputs.coalesce(1).write.mode('overwrite').option("header", "true").csv(output_path)
	shutil.rmtree(output_path)

	# pid = os.getpid()
	pid = file_id
	time_taken = time.time() - start_time
	print("Done with file {} on process {} in time {}".format(file_id, pid, time_taken))

	return pid, int(time_taken)

## fft function
@pandas_udf('seconds string, epochs integer, EEG double, EEG2 double, EMG double, EEG_f double, EEG2_f double, EMG_f double, score integer', PandasUDFType.GROUPED_MAP)
def sci_fft(pdf):
	detrended = detrend(pdf[['EEG','EEG2', 'EMG']], axis=0, type='constant')
	pdf['EEG_f'] = pd.DataFrame(1 / (30*62) * np.abs(fft(detrended[:,0])))
	pdf['EEG2_f'] = pd.DataFrame(1 / (30*62) * np.abs(fft(detrended[:,1])))
	pdf['EMG_f'] = pd.DataFrame(1 / (30*62) * np.abs(fft(detrended[:,2])))
	return pdf

if __name__ == '__main__':
	edf_filenames = [f for f in os.listdir(edf_dir) if os.path.isfile(edf_dir + f)]
	xml_filenames = [f for f in os.listdir(xml_dir) if os.path.isfile(xml_dir + f)]
	edf_filenames.sort()
	xml_filenames.sort()

	with open('../output/spark_time_metrics_all.csv', 'w') as f:
		# Open a new metrics file to write to after each epoch of training
		f.write("pid, time\n")
		f.close()

	serial_load()
