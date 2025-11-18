from kaggle import api

dataset_path = 'chenrich/kdef-database'
download_path = '../dataset/orig_kdef/'
api.dataset_download_files(dataset_path, path=download_path, unzip=True)