import os
import shutil
import numpy as np

"""
Original dataset has too much data. This randomly partitions the data into 
a smaller chunk and copies the files into a different location.
"""

def assert_match(edf_path, xml_path):
	"""
	Assert that the files in edf and xml directories match
	:param edf_path:
	:param xml_path:
	:return:
	"""
	edf_filenames = [f for f in os.listdir(edf_path) if os.path.isfile(edf_path + f)]  # list of all files in dir
	xml_filenames = [f for f in os.listdir(xml_path) if os.path.isfile(xml_path + f)]
	assert len(edf_filenames) == len(xml_filenames), \
		'Number of files in "{}" and "{}" do not match'.format(edf_path, xml_path)

	edf_filenames.sort()
	xml_filenames.sort()
	for i in range(0, len(edf_filenames)):
		assert edf_filenames[i].split('.')[0] == xml_filenames[i].split('.')[0][:-5], 'Some file(s) do not match'


def extract(src_edf_path, src_xml_path, dst_edf_path, dst_xml_path, frac_extract):
	"""
	Randomly partitions the dataset and copies a small fraction over into a new directory

	:param src_edf_path: str, directory path to xml src files (ending in '/')
	:param src_xml_path: str, directory path to edf src files (ending in '/')
	:param dst_edf_path: str, directory path to where the dst xml files should go (ending in '/')
	:param frac_extract: fraction of full dataset to extract
	:return: None
	"""
	# TODO error handling for if dst path already has files (delete existing?)
	# TODO error handling for if src/dst dirs don't exist

	src_edf_files = [f for f in os.listdir(src_edf_path) if os.path.isfile(src_edf_path + f)]  # list of all files in src dir
	src_xml_files = [f for f in os.listdir(src_xml_path) if os.path.isfile(src_xml_path + f)]
	src_edf_files.sort()
	src_edf_files.sort()

	shuffled_ix = np.random.permutation(len(src_edf_files))
	cutoff = int(frac_extract * len(shuffled_ix))
	for i in range(0, cutoff):
		shutil.copy(src_edf_path + src_edf_files[shuffled_ix[i]], dst_edf_path)
		shutil.copy(src_xml_path + src_xml_files[shuffled_ix[i]], dst_xml_path)


def main():
	data_src_edf = '../data/full/edfs/shhs1/'
	data_src_xml = '../data/full/annotations/shhs1/'
	data_dst_edf = '../data/5percent/edfs/shhs1/'
	data_dst_xml = '../data/5percent/annotations/shhs1/'
	frac_extract = 0.05

	assert_match(data_src_edf, data_src_xml)
	extract(data_src_edf, data_src_xml, data_dst_edf, data_dst_xml, frac_extract)
	assert_match(data_dst_edf, data_dst_xml)


if __name__ == "__main__":
	main()