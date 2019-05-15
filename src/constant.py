
### DIRECTORY STRUCTURE CONSTANTS ###
RAW_EDF_DIR   = '../data/5percent/edfs/shhs1/'         # EDF files (raw data)
RAW_XML_DIR   = '../data/5percent/annotations/shhs1/'  # XML files (raw data)
PROCESSED_DIR = '../data/5percent/processed/'          # CSV files containing preprocessed data
SPLITS_DIR    = './splits/'                            # TXT files containing what csv files are in train/valid/test set
GRAPHS_DIR = '../graphs/'                              # For image plots
OUTPUT_DIR = '../output/'                              # For text logs, and everything else



### PREPROCESSED DATA PROPERTIES ###
INPUT_SAMPLING_FREQ = 125   # Hertz (for EEG, EEG2, and EMG ONLY)
FINAL_SAMPLING_FREQ = 62
S_PER_EPOCH = 30


### MODEL TRAINING PROPERTIES ###
NUM_EPOCHS = 40
BATCH_SIZE = 32
USE_CUDA = True