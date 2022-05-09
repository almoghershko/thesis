# script params:
# run_name
# full_epoch (0/1)
# epochs
# sub_epochs
# snr_range_min_db
# snr_range_max_db

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
s3_data_ver_dir_path = os.path.join(s3_data_dir_path,'100K_V4')
save_RF_dir = 'simple___2022_05_07___18_57_07___100K_V4_training_set'
s3_urf_save_dir_path = os.path.join(s3_saves_dir_path, 'RF', save_RF_dir)
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

# load data

print('Loading data from data dir: {0}'.format(s3_data_ver_dir_path))
gs = from_s3_pkl(s3_client, bucket_name, os.path.join(s3_data_ver_dir_path,'gs_train_V4.pkl'))
wl_grid = from_s3_npy(s3_client, bucket_name, os.path.join(s3_data_ver_dir_path, 'wl_100K_V4.npy'))


print('Loading data from RF dir: {0}'.format(s3_urf_save_dir_path))
X = from_s3_npy(s3_client, bucket_name, os.path.join(s3_urf_save_dir_path, 'X.npy'))
#I_train = from_s3_npy(s3_client, bucket_name, os.path.join(s3_urf_save_dir_path, 'I_train.npy'))
dist_mat = from_s3_npy(s3_client, bucket_name, os.path.join(s3_urf_save_dir_path, 'dist_mat.npy'))

# # Siamese Networks

# In[18]:


import GPUtil
print('GPUs:\n{0}'.format('\n'.join(['('+str(i+1)+')\t'+gpu.name for i,gpu in enumerate(GPUtil.getGPUs())])))


# In[19]:


# Make sure a GPU is available
import tensorflow as tf
assert tf.config.list_physical_devices('GPU')[0].device_type == 'GPU', 'GPU is not available!'


# In[20]:


save_NN = True
save_NN_name = 'Kernels_31_Filters_64_32_16_8_4_Hiddens_512_128_tanh'

load_NN = False
load_NN_name = ''


# In[21]:


assert not (save_NN and load_NN), '"save" and "load" cant both be "True"'


# ## Creating the model

# In[22]:


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


# In[23]:


from NN import DistanceLayer, SiameseModel

# ====================================
# SIAMESE NETWORK
# ====================================

if not load_NN:
    
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
    x = layers.BatchNormalization()(x)
    x = activations.tanh(x)
    x_out = x

    # creating the model
    embedding = Model(x_in, x_out)
    embedding.summary()
    
    # ----------------------
    # Siamese network
    # ----------------------
    
    first_input = layers.Input(name="first_input", shape=(len(wl_grid)))
    second_input = layers.Input(name="second_input", shape=(len(wl_grid)))

    distance = DistanceLayer()(
        embedding(first_input),
        embedding(second_input)
    )

    siamese_network = Model(
        inputs=[first_input, second_input], outputs=distance
    )
    siamese_network.summary()
    
else:
    
    # building the load path
    load_NN_dir_path = os.path.join(saves_dir_path, 'NN', '100K', load_NN_name)
    model_path = os.path.join(load_NN_dir_path, 'model')
    
    # loading the saved model
    siamese_network = tf.keras.models.load_model(path, custom_objects={'DistanceLayer': DistanceLayer})

# ====================================
# SIAMESE MODEL
# ====================================
siamese_model = SiameseModel(siamese_network)
siamese_model.compile(optimizer=optimizers.Adam(0.001))


# ## Creating data loaders

# In[24]:

from sklearn.model_selection import train_test_split
if not load_NN:

    from NN import DistillationDataGenerator

    X_train, X_val, I_train, I_test = train_test_split(X, np.arange(X.shape[0]), train_size=9000, random_state=seed)

    batch_size = 128
    
    full_epoch = int(sys.argv[2])
    if len(sys.argv)>5:
        snr_range_db = [int(sys.argv[5]),int(sys.argv[6])]
    else:
        snr_range_db = None

    from NN import DistillationDataGenerator
    train_gen_full = DistillationDataGenerator(X_train, dist_mat[I_train,:][:,I_train], batch_size=batch_size, shuffle=True, seed=seed, snr_range_db=snr_range_db, full_epoch=full_epoch)
    train_gen = DistillationDataGenerator(X_train, dist_mat[I_train,:][:,I_train], batch_size=batch_size, shuffle=True, seed=seed, snr_range_db=snr_range_db, full_epoch=False)
    val_gen = DistillationDataGenerator(X_val, dist_mat[I_test,:][:,I_test], batch_size=batch_size, shuffle=True, seed=seed, snr_range_db=snr_range_db, full_epoch=False)


# ## Training the model

# In[25]:


if save_NN:
    
    # create a save dir
    from datetime import datetime
    run_prefix = os.environ['RUN_PREFIX']
    s3_save_NN_dir_path = os.path.join(s3_saves_dir_path, 'NN', '100K', run_prefix+'___' + datetime.now().strftime("%Y_%m_%d___%H_%M_%S") + '___' + save_NN_name)
    print('save NN folder (S3): ' + s3_save_NN_dir_path)


# In[26]:


if not load_NN:

    epochs = int(sys.argv[3]) # 5 # [full]
    sub_epochs = int(sys.argv[4]) # 1 # [full]
    hist_epochs = 10 # [not full]
    from progressbar import progressbar
    N = 1000
    edges = np.linspace(0,1,N+1)
    centers = (edges[:-1]+edges[1:])/2
    N_chunks = int(epochs/sub_epochs)
    loss_history = []
    val_loss_history = []
    
    # create the figures for the histogram and for the loss
    from matplotlib import pyplot as plt
    hist_fig, hist_ax = plt.subplots(figsize=(15,8))
    hist_ax.set_title('Loss histogram')
    hist_ax.set_xlabel('loss')
    hist_ax.set_ylabel('freq')
    hist_ax.grid()
    loss_fig, loss_ax = plt.subplots(figsize=(15,8))
    loss_ax.set_title('Training curve')
    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    loss_ax.grid()
    
    import time
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
            
        # calculate histogram
        print('calculating histogram')
        counts = np.zeros(shape=(N,), dtype=int)
        for _ in range(hist_epochs):
            for data in progressbar(train_gen):
                l = siamese_model._compute_loss(data)
                hist,_ = np.histogram(l, bins=edges)
                counts += hist
        counts = np.divide(counts, np.sum(counts))
        
        # plot the histogram
        curr_epochs = (i_chunk+1)*sub_epochs
        hist_ax.plot(centers[:150],counts[:150],label='{0} epochs'.format(curr_epochs))
        hist_ax.legend()
        
        # plot the loss
        e = np.arange(curr_epochs)+1
        loss_ax.plot(e, loss_history, label='training')
        loss_ax.plot(e, val_loss_history, label='test')
        loss_ax.legend()
        
        if save_NN:
            to_s3_fig(hist_fig, s3_client, bucket_name, os.path.join(s3_save_NN_dir_path, 'loss_hist_after_{0}_epochs.png'.format((i_chunk+1)*sub_epochs)))
            to_s3_fig(loss_fig, s3_client, bucket_name, os.path.join(s3_save_NN_dir_path, 'loss.png'))
            to_s3_npy(np.array(loss_history), s3_client, bucket_name, os.path.join(s3_save_NN_dir_path, 'loss.npy'))
            to_s3_npy(np.array(val_loss_history), s3_client, bucket_name, os.path.join(s3_save_NN_dir_path, 'val_loss.npy'))
            
    end_time = time.time()
    print('TOTAL TIME = {0:.3f} hours'.format((end_time - start_time)/3600))


# In[27]:


if not load_NN:

    # plot the loss
    from matplotlib import pyplot as plt
    fig,ax = plt.subplots(figsize=(15,8))
    e = np.arange(epochs)+1
    ax.plot(e, loss_history, label='training')
    ax.plot(e, val_loss_history, label='test')
    ax.set_title('Training curve')
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.legend()
    ax.grid()
    
    if save_NN:
        to_s3_fig(fig, s3_client, bucket_name, os.path.join(s3_save_NN_dir_path, 'loss.png'))
        ax.set_yscale('log')
        to_s3_fig(fig, s3_client, bucket_name, os.path.join(s3_save_NN_dir_path, 'loss_log.png'))


# In[28]:


if save_NN:

    # get model summary
    stringlist = []
    embedding.summary(print_fn=lambda x: stringlist.append(x))
    embedding_summary = "\n".join(stringlist)
    stringlist = []
    siamese_network.summary(print_fn=lambda x: stringlist.append(x))
    siamese_network_summary = "\n".join(stringlist)

    # save log
    from s3 import log_s3
    log_s3(s3_client, bucket_name, s3_save_NN_dir_path, 'NN_log.txt',
        s3_urf_save_dir_path = s3_urf_save_dir_path,
        embedding_summary = embedding_summary,
        siamese_network_summary = siamese_network_summary
        )
    
    # save the network
    s3_model_path = os.path.join(s3_save_NN_dir_path, 'model')
    from s3 import s3_save_TF_model
    s3_save_TF_model(siamese_model, s3_client, bucket_name, s3_model_path)

