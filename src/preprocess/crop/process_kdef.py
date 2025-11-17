import os
import sys

# Ensure the parent 'src' directory is on sys.path so we can import preprocess
CURRENT_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if SRC_DIR not in sys.path:
	sys.path.append(SRC_DIR)

from preprocess.img_preprocess import process_dataset_tree

input_directory = '../dataset/orig_kdef/'
output_directory = '../dataset/processed_kdef/'

# Process entire tree and keep subdirectories organized
process_dataset_tree(input_directory, output_directory, dim=(224, 224))


