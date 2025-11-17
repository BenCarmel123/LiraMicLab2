from preprocess import process_dataset

input_directory = '../dataset/orig_kdef/'
output_directory = '../dataset/processed_kdef/'

process_dataset(input_directory, output_directory, dim=(224, 224))


