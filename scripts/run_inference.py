# imports
import pandas as pd
import numpy as np
import os
import sys
import pickle
from matplotlib import pyplot as plt
import matplotlib
import boto3

# random seed
seed = 42
np.random.seed(seed)

# local files paths
local_home_dir_path = os.path.expanduser("~")
local_work_dir_path = os.path.join(local_home_dir_path, 'thesis')
local_code_dir_path = os.path.join(local_work_dir_path , 'code')

# S3 file paths
endpoint_url = 'https://s3-west.nrp-nautilus.io'
bucket_name = 'tau-astro'
prefix = 'almogh'
s3_work_dir_path = os.path.join(prefix, 'workdir3')
s3_saves_dir_path = os.path.join(s3_work_dir_path , 'model_saves')
s3_data_dir_path = os.path.join(s3_work_dir_path , 'data')
s3_data_ver_dir_path = os.path.join(s3_data_dir_path,'100K_V1')
save_RF_dir = 'small_URF_10K_train_set__2022_03_27___13_00_39'
s3_runs_dir_path = os.path.join(s3_work_dir_path , 'runs')

s3_client = boto3.client("s3", endpoint_url=endpoint_url)

# adding code folder to path
sys.path.insert(1, local_code_dir_path)
from s3 import to_s3_npy, to_s3_pkl, from_s3_npy, from_s3_pkl, to_s3_fig

# get run directory
run_name = sys.argv[1]
s3_run_dir_path = os.path.join(s3_runs_dir_path, run_name)
print('run dir path = {0}'.format(s3_run_dir_path))

# load data from the data dir
gs = from_s3_pkl(s3_client, bucket_name, os.path.join(s3_data_ver_dir_path,'gs.pkl'))
X = from_s3_npy(s3_client, bucket_name, os.path.join(s3_data_ver_dir_path, 'spec.npy'))
wl_grid = from_s3_npy(s3_client, bucket_name, os.path.join(s3_data_ver_dir_path, 'wl_grid.npy'))

# load the data from the URF dir
s3_urf_save_dir_path = os.path.join(s3_saves_dir_path, 'RF', save_RF_dir)
print('RF save folder (S3): ' + s3_urf_save_dir_path)
I_train = from_s3_npy(s3_client, bucket_name, os.path.join(s3_urf_save_dir_path, 'I_train.npy'))
X = X[I_train]
dist_mat = from_s3_npy(s3_client, bucket_name, os.path.join(s3_urf_save_dir_path, 'dist_mat.npy'))

# create data loaders
batch_size = 128
from NN import DistillationDataGenerator
train_gen = DistillationDataGenerator(X, dist_mat, shuffle=False, seed=seed, batch_size=batch_size, full_epoch=True)
val_gen = DistillationDataGenerator(X, dist_mat, shuffle=False, seed=seed, batch_size=batch_size, full_epoch=True)

from s3 import s3_load_TF_model
from NN import DistanceLayer
siamese_model = s3_load_TF_model(s3_client,
                                 bucket_name=bucket_name,
                                 path_in_bucket='almogh/workdir3/model_saves/NN/100K/LongTrainDenoise___2022_04_02___07_30_13___Kernels_31_Filters_64_32_16_8_4_Hiddens_512_128_tanh/model',
                                 model_name='model',
                                 custom_objects={'DistanceLayer': DistanceLayer})




