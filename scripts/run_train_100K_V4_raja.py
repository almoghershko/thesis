# script params:
# run_name
# full_epoch (0/1)
# epochs
# sub_epochs
# snr_range_min_db
# snr_range_max_db

save_NN_name = 'Kernels_31_Filters_64_32_16_8_4_Hiddens_512_128_tanh_sigmoidDist'

# imports
import pandas as pd
import numpy as np
import os
import sys
import pickle
from matplotlib import pyplot as plt
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
from NN import DistanceLayer, SiameseModel, DistillationDataGenerator, NormalizeDistance

# get run directory
run_name = sys.argv[1]
s3_run_dir_path = os.path.join(s3_runs_dir_path, run_name)
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
from tensorflow.linalg import normalize

tf.random.set_seed(seed)

# ----------------------
# Embedding Network
# ----------------------

hidden_size = 512
embedding_size = 128

# input layer
x_in = layers.Input(shape=(len(wl_grid), 1))

# adding the network layers
x = x_in
#
x = layers.Conv1D(64, 31, activation=None, padding='same',
                  kernel_initializer=initializers.GlorotUniform(seed=seed))(x)
x = layers.BatchNormalization()(x)
x = activations.relu(x)
x = layers.AveragePooling1D( 2, padding='same')(x)
#
x = layers.Conv1D(32, 31, activation=None, padding='same',
                  kernel_initializer=initializers.GlorotUniform(seed=seed))(x)
x = layers.BatchNormalization()(x)
x = activations.relu(x)
x = layers.AveragePooling1D( 2, padding='same')(x)
#
x = layers.Conv1D(16, 31, activation=None, padding='same',
                  kernel_initializer=initializers.GlorotUniform(seed=seed))(x)
x = layers.BatchNormalization()(x)
x = activations.relu(x)
x = layers.AveragePooling1D( 2, padding='same')(x)
#
x = layers.Conv1D(8, 31, activation=None, padding='same',
                  kernel_initializer=initializers.GlorotUniform(seed=seed))(x)
x = layers.BatchNormalization()(x)
x = activations.relu(x)
x = layers.AveragePooling1D( 2, padding='same')(x)
#
x = layers.Conv1D(4, 31, activation=None, padding='same',
                  kernel_initializer=initializers.GlorotUniform(seed=seed))(x)
x = layers.BatchNormalization()(x)
x = activations.relu(x)
x = layers.AveragePooling1D( 2, padding='same')(x)
#
x = layers.Flatten()(x)
x = layers.Dense(hidden_size,
                kernel_initializer=initializers.GlorotUniform(seed=seed))(x)
x = layers.BatchNormalization()(x)
x = activations.relu(x)
x = layers.Dense(embedding_size,
                kernel_initializer=initializers.GlorotUniform(seed=seed))(x)
#x = layers.BatchNormalization()(x)
#x = activations.tanh(x)

x = normalize(x, axis=1) # default is Euqlidean norm

x_out = x[1]

# creating the model
embedding = Model(x_in, x_out, name="embedding")
embedding.summary()

# ----------------------
# Siamese network
# ----------------------

first_input = layers.Input(name="first_input", shape=(len(wl_grid)))
second_input = layers.Input(name="second_input", shape=(len(wl_grid)))  

first_embedding = embedding(first_input)
second_embedding = embedding(second_input)

#distance = DistanceLayer()(
#    first_embedding,
#    second_embedding
#)
#distance = activations.sigmoid(distance)

dot = layers.Dot(axes=1)([first_embedding, second_embedding]) # dot product should be in range [-1,1]
distance = NormalizeDistance()(dot)

siamese_network = Model(
    inputs=[first_input, second_input], outputs=distance, name="siamese model"
)
siamese_network.summary()

# ----------------------
# Siamese model
# ----------------------
siamese_model = SiameseModel(siamese_network)
siamese_model.compile(optimizer=optimizers.Adam(0.001))

# ===================================
# Training the model
# ===================================

X_train, X_val, I_train, I_test = train_test_split(X, np.arange(X.shape[0]), train_size=0.9, random_state=seed)

batch_size = 128

full_epoch = int(sys.argv[2])
if len(sys.argv)>5:
    snr_range_db = [int(sys.argv[5]),int(sys.argv[6])]
    awgn_str = 'Adding AWGN with SNR uniformly distributed in range [{0},{1}] dB'.format(snr_range_db[0],snr_range_db[1])
else:
    snr_range_db = None
    awgn_str = 'Not adding AWGN'
print('=====================================================================================')
print(awgn_str) 
print('=====================================================================================')

train_gen_full = DistillationDataGenerator(X_train, dist_mat[I_train,:][:,I_train], batch_size=batch_size, shuffle=True, seed=seed, snr_range_db=snr_range_db, full_epoch=full_epoch)
train_gen = DistillationDataGenerator(X_train, dist_mat[I_train,:][:,I_train], batch_size=batch_size, shuffle=True, seed=seed, snr_range_db=snr_range_db, full_epoch=False)
val_gen = DistillationDataGenerator(X_val, dist_mat[I_test,:][:,I_test], batch_size=batch_size, shuffle=True, seed=seed, snr_range_db=snr_range_db, full_epoch=False)

# ===================================
# Training the model
# ===================================

# create a save dir
run_prefix = os.environ['RUN_PREFIX']
s3_save_NN_dir_path = os.path.join(s3_saves_dir_path, 'NN', data_ver, run_prefix+'___' + datetime.now().strftime("%Y_%m_%d___%H_%M_%S") + '___' + save_NN_name)
save_dir_str = 'save NN folder (S3): ' + s3_save_NN_dir_path
print('=====================================================================================')
print(save_dir_str)
print('=====================================================================================')

epochs = int(sys.argv[3]) # 5 # [full]
sub_epochs = int(sys.argv[4]) # 1 # [full]
hist_epochs = 10 # [not full]
N_chunks = int(epochs/sub_epochs)
loss_history = []
val_loss_history = []

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

# training loop
training_str = 'Training for {0} {1} epochs, and stopping for saving every {2} {1} epochs, for a total of {3} stages.'.format(epochs, 'full' if full_epoch else 'partial', sub_epochs, N_chunks)
print('=====================================================================================')
print(training_str)
print('=====================================================================================')
start_time = time.time()
for i_chunk in range(N_chunks):
    
    print('-------------------------------------')
    print('epochs {0}-{1}:'.format(i_chunk*sub_epochs+1, (i_chunk+1)*sub_epochs))
    print('-------------------------------------')

    # train
    try:
        # for some reason, the first call to fit will throw KeyError...
        history = siamese_model.fit(train_gen_full, epochs=sub_epochs, validation_data=val_gen, verbose=2)
    except KeyError:
        history = siamese_model.fit(train_gen_full, epochs=sub_epochs, validation_data=val_gen, verbose=2)
    loss_history += history.history['loss']
    val_loss_history += history.history['val_loss']
    
    # plot the loss
    curr_epochs = (i_chunk+1)*sub_epochs
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
    s3_save_NN_dir_path_sub_epoch = os.path.join(s3_save_NN_dir_path, 'after_{0}_epochs'.format((i_chunk+1)*sub_epochs))
    # save the figures
    to_s3_fig(loss_fig, s3_client, bucket_name, os.path.join(s3_save_NN_dir_path_sub_epoch, 'loss.png'))
    to_s3_fig(log_loss_fig, s3_client, bucket_name, os.path.join(s3_save_NN_dir_path_sub_epoch, 'loss.png'))
    # save the losses
    to_s3_npy(np.array(loss_history), s3_client, bucket_name, os.path.join(s3_save_NN_dir_path_sub_epoch, 'loss.npy'))
    to_s3_npy(np.array(val_loss_history), s3_client, bucket_name, os.path.join(s3_save_NN_dir_path_sub_epoch, 'val_loss.npy'))
    # get model summary
    stringlist = []
    embedding.summary(print_fn=lambda x: stringlist.append(x))
    embedding_summary = "\n".join(stringlist)
    stringlist = []
    siamese_network.summary(print_fn=lambda x: stringlist.append(x))
    siamese_network_summary = "\n".join(stringlist)
    # save log
    log_s3(s3_client, bucket_name, s3_save_NN_dir_path_sub_epoch, 'NN_log.txt',
        s3_NN_save_dir_path = save_dir_str,
        s3_urf_save_dir_path = s3_urf_save_dir_path,
        epochs = training_str,
        training_duration = time_str,
        awgn = awgn_str,
        embedding_summary = embedding_summary,
        siamese_network_summary = siamese_network_summary
        )
    # save the network
    s3_model_path = os.path.join(s3_save_NN_dir_path_sub_epoch, 'model')
    s3_save_TF_model(siamese_model, s3_client, bucket_name, s3_model_path)

