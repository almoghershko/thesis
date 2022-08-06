# get script parameters
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("run_name", help="the name for the run", type=str)
parser.add_argument("model_dir", help="the name for directory of the NN save", type=str)
parser.add_argument("epochs", help="number of epochs to take from the save dir", type=int)
parser.add_argument("--snr_min", help="minimal snr", default=5, type=int)
parser.add_argument("--snr_max", help="maximal snr", default=30, type=int)
parser.add_argument("--snr_step", help="snr step", default=5, type=int)
args = parser.parse_args()

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

s3_client = boto3.client("s3", endpoint_url=endpoint_url)

# adding code folder to path
sys.path.insert(1, local_code_dir_path)
s3_runs_dir_path = os.path.join(s3_work_dir_path , 'runs')
from s3 import to_s3_npy, to_s3_pkl, from_s3_npy, from_s3_pkl, to_s3_fig

# get run directory
s3_run_dir_path = os.path.join(s3_runs_dir_path, args.run_name)
print('run dir path = {0}'.format(s3_run_dir_path))

# =========================
# LOAD DATA
# =========================

# load data
print('Loading data and creating dataset')
gs_100K = from_s3_pkl(s3_client, bucket_name, os.path.join(s3_v4_data_ver_dir_path,'gs_100K_V4.pkl'))
gs_train = from_s3_pkl(s3_client, bucket_name, os.path.join(s3_v4_data_ver_dir_path,'gs_train_V4.pkl'))
gs_test = from_s3_pkl(s3_client, bucket_name, os.path.join(s3_v4_data_ver_dir_path,'gs_test_V4.pkl'))
full_wl_grid = from_s3_npy(s3_client, bucket_name, os.path.join(s3_data_dir_path, 'wl_grid.npy'))
wl_grid = from_s3_npy(s3_client, bucket_name, os.path.join(s3_v4_data_ver_dir_path, 'wl_100K_V4.npy'))

load_small_RF_name = 'simple___2022_05_07___18_57_07___100K_V4_training_set'
s3_load_small_dir_path = os.path.join(s3_saves_dir_path, 'RF', load_small_RF_name)
X = from_s3_npy(s3_client, bucket_name, os.path.join(s3_load_small_dir_path, 'X.npy'))

I_sort = np.argsort(gs_train['snMedian'])[::-1][:1000]
X = X[I_sort,:]

# =========================
# LOAD AND INFER BIG RF
# =========================

load_RF_name = 'simple___2022_05_10___11_24_58___100K_V4_full_data_set'
s3_load_dir_path = os.path.join(s3_saves_dir_path, 'RF', load_RF_name)
print('loading from folder (S3): {0}'.format(s3_load_dir_path))

from CustomRandomForest import CustomRandomForest
rf = CustomRandomForest.load_s3(s3_client, bucket_name, os.path.join(s3_load_dir_path, 'crf.pkl'))

print('Applying the RF (calculate leaves)')
X_leaves = rf.apply(X)

print('Predicting fully')
Y_hat = rf.predict_full_from_leaves(X_leaves)

print('Calculating the similarity matrix')
from CustomRandomForest import build_similarity_matrix
sim_mat = build_similarity_matrix(X_leaves, Y_hat)

print('Calculating the distance matrix and weirdness scores')
D_RF = 1 - sim_mat
weird_scores_RF = np.mean(D_RF, axis=1)

# =========================
# LOAD SMALL RF
# =========================

print('loading from folder (S3): {0}'.format(s3_load_small_dir_path))

from CustomRandomForest import CustomRandomForest
small_rf = CustomRandomForest.load_s3(s3_client, bucket_name, os.path.join(s3_load_small_dir_path, 'crf.pkl'))

# =========================
# LOAD NN
# =========================

from s3 import s3_load_TF_model
from NN import DistanceLayer

denosie_model_save_dir_path = os.path.join(s3_saves_dir_path,'NN','100K_V4',args.model_dir,"after_{0}_epochs".format(args.epochs))
siamese_denoise_model = s3_load_TF_model(s3_client,
                                 bucket_name=bucket_name,
                                 path_in_bucket=os.path.join(denosie_model_save_dir_path,'model'),
                                 model_name='model',
                                 custom_objects={'DistanceLayer': DistanceLayer})
                                 
batch_size = 128
from NN import DistillationDataGenerator

# =========================
# Checking Error VS SNR
# =========================

SNRs = [snr for snr in range(args.snr_min,args.snr_max,args.snr_step)] + [100]

D_RF_snr = np.zeros(shape=(len(SNRs),D_RF.shape[0],D_RF.shape[1]))
D_small_RF_snr = np.zeros(shape=(len(SNRs),D_RF.shape[0],D_RF.shape[1]))
#D_NN_snr = np.zeros(shape=(len(SNRs),D_RF.shape[0],D_RF.shape[1]))
D_NN_denoise_snr = np.zeros(shape=(len(SNRs),D_RF.shape[0],D_RF.shape[1]))

for i,snr in enumerate(SNRs):
    
    print('snr = {0}'.format(snr))
    
    print('creating a noisy copy')
    x = X.copy()
    N_std_x = np.sqrt((np.std(x,axis=1)**2)/snr)
    x += N_std_x.reshape(-1,1)*np.random.randn(x.shape[0],x.shape[1])
    
    print('RF inference')
    X_leaves_snr = rf.apply(x)
    Y_hat_snr = rf.predict_full_from_leaves(X_leaves_snr)
    sim_mat_snr = build_similarity_matrix(X_leaves_snr, Y_hat_snr)
    D_RF_snr[i] = 1 - sim_mat_snr
    
    print('small RF inference')
    X_leaves_snr_small = small_rf.apply(x)
    Y_hat_snr_small = small_rf.predict_full_from_leaves(X_leaves_snr_small)
    sim_mat_snr_small = build_similarity_matrix(X_leaves_snr_small, Y_hat_snr_small)
    D_small_RF_snr[i] = 1 - sim_mat_snr_small
    
    # normalizing the spectra for the NN
    x_min = x.min(axis=1).reshape(-1,1)
    x -= x_min
    x_max = x.max(axis=1).reshape(-1,1)
    x /= x_max
    """
    print('NN inference')
    data_gen = DistillationDataGenerator(x, np.zeros(shape=(x.shape[0], x.shape[0])), shuffle=False, seed=seed, batch_size=batch_size, full_epoch=True)
    Z_NN = siamese_model.predict(data_gen, verbose=1)
    N = int((-1+np.sqrt(1+8*len(Z_NN)))/2)
    D_NN = np.zeros(shape=(N,N))
    D_NN[np.triu_indices(N)] = Z_NN
    D_NN = D_NN.T
    D_NN[np.triu_indices(N)] = Z_NN
    D_NN_snr[i] = D_NN
    """
    print('denoise NN inference')
    data_gen = DistillationDataGenerator(x, np.zeros(shape=(x.shape[0], x.shape[0])), shuffle=False, seed=seed, batch_size=batch_size, full_epoch=True)
    Z_NN = siamese_denoise_model.predict(data_gen, verbose=1)
    N = int((-1+np.sqrt(1+8*len(Z_NN)))/2)
    D_NN = np.zeros(shape=(N,N))
    D_NN[np.triu_indices(N)] = Z_NN
    D_NN = D_NN.T
    D_NN[np.triu_indices(N)] = Z_NN
    D_NN_denoise_snr[i] = D_NN

# =========================
# PLOTTING
# =========================

# truncating the NN distance matrices
D_NN_denoise_snr_truc = D_NN_denoise_snr.copy()
D_NN_denoise_snr_truc[D_NN_denoise_snr_truc>1] = 1

RF_err = [np.mean(np.abs(D_RF_snr[i]-D_RF)) for i in range(len(SNRs))]
small_RF_err = [np.mean(np.abs(D_small_RF_snr[i]-D_RF)) for i in range(len(SNRs))]
#NN_err = [np.mean(np.abs(D_NN_snr_truc[i]-D_RF)) for i in range(len(SNRs))]
NN_denoise_err = [np.mean(np.abs(D_NN_denoise_snr_truc[i]-D_RF)) for i in range(len(SNRs))]

fig,ax = plt.subplots(figsize=(12,8))
ax.plot(SNRs, RF_err, label='RF')
ax.plot(SNRs, small_RF_err, label='small RF')
#ax.plot(SNRs, NN_err, label='NN')
ax.plot(SNRs, NN_denoise_err, label='NN denoise')
ax.legend()
plt.grid()

to_s3_fig(fig, s3_client, bucket_name, os.path.join(s3_run_dir_path, 'err_vs_snr.png'))