# get script parameters
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("run_name", help="the name for the run", type=str)
parser.add_argument("epochs", help="number of epochs", type=int)
parser.add_argument("--short_epochs", help="epoch is not a full epoch", action="store_true")
parser.add_argument("--sub_epochs", help="number of epochs for save intervals", default=1, type=int)
parser.add_argument("--snr_min", help="the minimal SNR (linear)", type=int)
parser.add_argument("--snr_max", help="the maximal SNR (linear)", type=int)
parser.add_argument("--norm_inputs", help="inputs to the model are normalized to [0,1]", action="store_true")
parser.add_argument("--dist_loss", help="The loss function for the distance", type=str, choices=['L1','L2'], default='L1')
parser.add_argument("--step", help="training step", default=0.001)
args = parser.parse_args()
print('args:\n\t'+'\n\t'.join(f'{k} = {v}' for k, v in vars(args).items()))

if args.snr_min!=None and args.snr_max!=None:
    snr_range = [args.snr_min, args.snr_max]
else:
    if args.snr_min==None and args.snr_max==None:
        snr_range = None
    else:
        raise Exception("received only one of snr_min/snr_max arguments")
    
# imports
import sys
import pandas as pd
import numpy as np
import os
import pickle
import matplotlib
from matplotlib import pyplot as plt
import boto3
import time
from datetime import datetime
from progressbar import progressbar
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
s3_saves_dir_path = os.path.join(s3_work_dir_path , 'model_saves')
s3_data_dir_path = os.path.join(s3_work_dir_path , 'data')
data_ver = '100K_V4'
s3_data_ver_dir_path = os.path.join(s3_data_dir_path,data_ver)
save_RF_dir = 'simple___2022_05_07___18_57_07___100K_V4_training_set'
s3_urf_save_dir_path = os.path.join(s3_saves_dir_path, 'RF', save_RF_dir)
s3_runs_dir_path = os.path.join(s3_work_dir_path , 'runs')

s3_client = boto3.client("s3", endpoint_url=endpoint_url)

# adding code folder to path, and importing the code
sys.path.insert(1, local_code_dir_path)
from s3 import to_s3_npy, to_s3_pkl, from_s3_npy, from_s3_pkl, to_s3_fig
from s3 import log_s3, s3_save_TF_model
from NN import DistanceLayer, SiameseModel, DistillationDataGenerator

# get run directory
s3_run_dir_path = os.path.join(s3_runs_dir_path, args.run_name)
print('run dir path = {0}'.format(s3_run_dir_path))

# ===================================
# Loading the data
# ===================================

# load data
print('Loading data from data dir: {0}'.format(s3_data_ver_dir_path))
gs = from_s3_pkl(s3_client, bucket_name, os.path.join(s3_data_ver_dir_path,'gs_train_V4.pkl'))
wl_grid = from_s3_npy(s3_client, bucket_name, os.path.join(s3_data_ver_dir_path, 'wl_100K_V4.npy'))

print('Loading data from RF dir: {0}'.format(s3_urf_save_dir_path))
X = from_s3_npy(s3_client, bucket_name, os.path.join(s3_urf_save_dir_path, 'X.npy'))
#I_train = from_s3_npy(s3_client, bucket_name, os.path.join(s3_urf_save_dir_path, 'I_train.npy'))
dist_mat = from_s3_npy(s3_client, bucket_name, os.path.join(s3_urf_save_dir_path, 'dis_mat.npy'))

# ===================================
# Creating the model
# ===================================

from tensorflow.keras import applications
from tensorflow.keras import activations
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from tensorflow.keras import Model
from tensorflow.keras import utils
from tensorflow.keras import initializers

tf.random.set_seed(seed)

# ----------------------
# Embedding Network
# ----------------------

# input layer
x_in = layers.Input(shape=(len(wl_grid), 1))

# adding the network layers
x = x_in
#
x = layers.Conv1D(filters=4 ,kernel_size=8,activation='relu',padding='same',kernel_initializer=initializers.HeNormal(seed=seed))(x)
x = layers.Conv1D(filters=16,kernel_size=8,activation='relu',padding='same',kernel_initializer=initializers.HeNormal(seed=seed))(x)
x = layers.MaxPooling1D( 4, padding='same')(x)
x = layers.Flatten()(x)
x = layers.Dense(256,kernel_initializer=initializers.HeNormal(seed=seed))(x)
x = layers.Dense(128,kernel_initializer=initializers.HeNormal(seed=seed))(x)
x = activations.tanh(x)
x_out = x

# creating the model
encoding = Model(x_in, x_out)
encoding.summary()

# ----------------------
# Siamese network
# ----------------------

first_input = layers.Input(name="first_input", shape=(len(wl_grid)))
second_input = layers.Input(name="second_input", shape=(len(wl_grid)))

first_encoding = encoding(first_input)
second_encoding = encoding(second_input)

distance = tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(first_encoding - second_encoding), -1),1e-9))

siamese_network = Model(
    inputs=[first_input, second_input], outputs=distance
)
siamese_network.summary()

# ----------------------
# Siamese model
# ----------------------
siamese_model = SiameseModel(siamese_network, dist_loss=args.dist_loss)
siamese_model.compile(optimizer=optimizers.Adam(args.step))

# ===================================
# Training the model
# ===================================

X_train, X_val, I_train, I_test = train_test_split(X, np.arange(X.shape[0]), train_size=0.9, random_state=seed)

batch_size = 128

train_gen_full = DistillationDataGenerator(X_train, dist_mat[I_train,:][:,I_train], batch_size=batch_size, shuffle=True, seed=seed, snr_range=snr_range, full_epoch=not(args.short_epochs), norm=args.norm_inputs)
val_gen = DistillationDataGenerator(X_val, dist_mat[I_test,:][:,I_test], batch_size=batch_size, shuffle=True, seed=seed, snr_range=snr_range, full_epoch=False, norm=args.norm_inputs)

# ===================================
# Training the model
# ===================================

# create a save dir
s3_save_NN_dir_path = os.path.join(s3_saves_dir_path, 'NN', data_ver, args.run_name)
print('save NN folder (S3): ' + s3_save_NN_dir_path)

N_chunks = int(args.epochs/args.sub_epochs)
loss_history = []
val_loss_history = []

# training loop
print('Training for {0} {1} epochs, and stopping for saving every {2} {1} epochs, for a total of {3} stages.'.format(args.epochs, 'short' if args.short_epochs else 'full', args.sub_epochs, N_chunks))
start_time = time.time()
for i_chunk in range(N_chunks):
    
    print('-------------------------------------')
    print('epochs {0}-{1}:'.format(i_chunk*args.sub_epochs+1, (i_chunk+1)*args.sub_epochs))
    print('-------------------------------------')

    # train
    try:
        # for some reason, the first call to fit will throw KeyError...
        history = siamese_model.fit(train_gen_full, epochs=args.sub_epochs, validation_data=val_gen, verbose=2)
    except KeyError:
        history = siamese_model.fit(train_gen_full, epochs=args.sub_epochs, validation_data=val_gen, verbose=2)
    loss_history += history.history['loss']
    val_loss_history += history.history['val_loss']
    
    # create the figures for the loss
    loss_fig, loss_ax = plt.subplots(figsize=(15,8))
    loss_ax.set_title('Training curve')
    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    loss_ax.grid()
    log_loss_fig, log_loss_ax = plt.subplots(figsize=(15,8))
    log_loss_ax.set_title('Training curve (Log Scale)')
    log_loss_ax.set_xlabel('epoch')
    log_loss_ax.set_ylabel('log(loss)')
    log_loss_ax.grid()
    log_loss_ax.set_yscale('log')
    
    # plot the loss
    curr_epochs = (i_chunk+1)*args.sub_epochs
    e = np.arange(curr_epochs)+1
    loss_ax.plot(e, loss_history, label='training')
    loss_ax.plot(e, val_loss_history, label='test')
    loss_ax.legend()
    log_loss_ax.plot(e, loss_history, label='training')
    log_loss_ax.plot(e, val_loss_history, label='test')
    log_loss_ax.legend()
    
    end_time = time.time()
    time_str = 'TOTAL TIME = {0:.3f} hours'.format((end_time - start_time)/3600)
    print(time_str)
    
    # create a sub dir
    s3_save_NN_dir_path_sub_epoch = os.path.join(s3_save_NN_dir_path, 'after_{0}_epochs'.format((i_chunk+1)*args.sub_epochs))
    # save the figures
    to_s3_fig(loss_fig, s3_client, bucket_name, os.path.join(s3_save_NN_dir_path_sub_epoch, 'loss.png'))
    to_s3_fig(log_loss_fig, s3_client, bucket_name, os.path.join(s3_save_NN_dir_path_sub_epoch, 'loss.png'))
    # save the losses
    to_s3_npy(np.array(loss_history), s3_client, bucket_name, os.path.join(s3_save_NN_dir_path_sub_epoch, 'loss.npy'))
    to_s3_npy(np.array(val_loss_history), s3_client, bucket_name, os.path.join(s3_save_NN_dir_path_sub_epoch, 'val_loss.npy'))
    # get model summary
    stringlist = []
    encoding.summary(print_fn=lambda x: stringlist.append(x))
    encoding_summary = "\n".join(stringlist)
    stringlist = []
    siamese_network.summary(print_fn=lambda x: stringlist.append(x))
    siamese_network_summary = "\n".join(stringlist)
    # save log
    log_s3(s3_client, bucket_name, s3_save_NN_dir_path_sub_epoch, 'NN_log.txt',
        args = '\n\t'+'\n\t'.join(f'{k} = {v}' for k, v in vars(args).items()),
        s3_save_NN_dir_path = s3_save_NN_dir_path,
        s3_urf_save_dir_path = s3_urf_save_dir_path,
        training_duration = time_str,
        encoding_summary = encoding_summary,
        siamese_network_summary = siamese_network_summary
        )
    # save the network
    s3_model_path = os.path.join(s3_save_NN_dir_path_sub_epoch, 'model')
    s3_save_TF_model(siamese_model, s3_client, bucket_name, s3_model_path)

