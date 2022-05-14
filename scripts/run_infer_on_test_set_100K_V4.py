# script params:
# run_name
# NN_save_name

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
s3_v2_data_ver_dir_path = os.path.join(s3_data_dir_path,'100K_V2')
s3_v4_data_ver_dir_path = os.path.join(s3_data_dir_path,'100K_V4')
s3_runs_dir_path = os.path.join(s3_work_dir_path , 'runs')

s3_client = boto3.client("s3", endpoint_url=endpoint_url)

# adding code folder to path
sys.path.insert(1, local_code_dir_path)
from s3 import to_s3_npy, to_s3_pkl, from_s3_npy, from_s3_pkl, to_s3_fig

# get run directory
run_name = sys.argv[1]
s3_run_dir_path = os.path.join(s3_runs_dir_path, run_name)
print('run dir path = {0}'.format(s3_run_dir_path))

# # Loading the data

print('Loading data and creating dataset')
gs = from_s3_pkl(s3_client, bucket_name, os.path.join(s3_v4_data_ver_dir_path,'gs_test_V4.pkl'))
X = from_s3_npy(s3_client, bucket_name, os.path.join(s3_v2_data_ver_dir_path, 'spec.npy'))
full_wl_grid = from_s3_npy(s3_client, bucket_name, os.path.join(s3_data_dir_path, 'wl_grid.npy'))
wl_grid = from_s3_npy(s3_client, bucket_name, os.path.join(s3_v4_data_ver_dir_path, 'wl_100K_V4.npy'))
start_i = (np.abs(full_wl_grid - wl_grid[0])).argmin()
end_i = 1+(np.abs(full_wl_grid - wl_grid[-1])).argmin()
X = X[gs.index, start_i:end_i]

# # Create data generator

batch_size = 128
from NN import DistillationDataGenerator
test_gen = DistillationDataGenerator(X, np.zeros(shape=(X.shape[0], X.shape[0])), shuffle=False, seed=seed, batch_size=batch_size, full_epoch=True)

# # Loading model

from s3 import s3_load_TF_model
from NN import DistanceLayer
NN_save_name = sys.argv[2]
model_save_dir_path = os.path.join(s3_saves_dir_path,'NN','100K_V4',NN_save_name)
siamese_model = s3_load_TF_model(s3_client,
                                 bucket_name=bucket_name,
                                 path_in_bucket=os.path.join(model_save_dir_path,'model'),
                                 model_name='model',
                                 custom_objects={'DistanceLayer': DistanceLayer})

siamese_model.summary()
siamese_model.siamese_network.summary()

# # Infering
dist_mat_hat_test_set = siamese_model.predict(test_gen, verbose=0)

# # Saving
to_s3_npy(dist_mat_hat_test_set, s3_client, bucket_name, os.path.join(model_save_dir_path,'dist_mat_hat_test_set.npy'))