# get script parameters
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("save_name", help="the name of the save", type=str)
parser.add_argument("epochs", help="number of epochs", type=int)
args = parser.parse_args()
print('args:\n\t'+'\n\t'.join(f'{k} = {v}' for k, v in vars(args).items()))

# imports
import sys
import pandas as pd
import numpy as np
import os
import pickle
import boto3
from sklearn.model_selection import train_test_split

# Make sure a GPU is available
import GPUtil
print('GPUs:\n{0}'.format('\n'.join(['('+str(i+1)+')\t'+gpu.name for i,gpu in enumerate(GPUtil.getGPUs())])))
import tensorflow as tf
assert tf.config.list_physical_devices('GPU')[0].device_type == 'GPU', 'GPU is not available!'

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
s3_runs_dir_path = os.path.join(s3_work_dir_path , 'runs')
s3_saves_dir_path = os.path.join(s3_work_dir_path , 'model_saves')
s3_data_dir_path = os.path.join(s3_work_dir_path , 'data')
data_ver = '100K_V4'
s3_data_ver_dir_path = os.path.join(s3_data_dir_path,data_ver)

# create S3 client
s3_client = boto3.client("s3", endpoint_url=endpoint_url)

# adding code folder to path, and importing the code
sys.path.insert(1, local_code_dir_path)

# load the data
from s3 import *
load_small_RF_name = 'simple___2022_05_07___18_57_07___100K_V4_training_set'
s3_load_small_dir_path = os.path.join(s3_saves_dir_path, 'RF', load_small_RF_name)
X = from_s3_npy(s3_client, bucket_name, os.path.join(s3_load_small_dir_path, 'X.npy'))

# load model
from NN import DistanceLayer, DistillationDataGenerator
denosie_model_save_dir_path = os.path.join(s3_saves_dir_path,'NN','100K_V4',args.save_name,'after_{0}_epochs'.format(args.epochs))
siamese_denoise_model = s3_load_TF_model(s3_client,
                                 bucket_name=bucket_name,
                                 path_in_bucket=os.path.join(denosie_model_save_dir_path,'model'),
                                 model_name='model',
                                 custom_objects={'DistanceLayer': DistanceLayer})

# predict
batch_size = 128
data_gen = DistillationDataGenerator(X, np.zeros(shape=(X.shape[0], X.shape[0])), shuffle=False, seed=seed, batch_size=batch_size, full_epoch=True, norm=True)
Z_NN = siamese_denoise_model.predict(data_gen, verbose=2)
N = int((-1+np.sqrt(1+8*len(Z_NN)))/2)
D_NN = np.zeros(shape=(N,N))
D_NN[np.triu_indices(N)] = Z_NN
D_NN = D_NN.T
D_NN[np.triu_indices(N)] = Z_NN

# save the distance matrix
to_s3_npy(D_NN, s3_client, bucket_name, os.path.join(denosie_model_save_dir_path, 'dist_mat.npy'))

