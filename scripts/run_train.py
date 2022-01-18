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
endpoint_url = 'https://s3.nautilus.optiputer.net'
bucket_name = 'tau-astro'
prefix = 'almogh'
s3_work_dir_path = os.path.join(prefix, 'workdir3')
s3_saves_dir_path = os.path.join(s3_work_dir_path , 'model_saves')
s3_data_dir_path = os.path.join(s3_work_dir_path , 'data')
s3_data_ver_dir_path = os.path.join(s3_data_dir_path,'HighSNR_12K_V1')
s3_runs_dir_path = os.path.join(s3_work_dir_path , 'runs')

s3_client = boto3.client("s3", endpoint_url=endpoint_url)

# adding code folder to path
sys.path.insert(1, local_code_dir_path)
from s3 import to_s3_npy, to_s3_pkl, from_s3_npy, from_s3_pkl, to_s3_fig

# get run directory
run_name = sys.argv[1]
s3_run_dir_path = os.path.join(s3_runs_dir_path, run_name)
print('run dir path = {0}'.format(s3_run_dir_path))


# # Train RF

# In[3]:


save_RF = False
save_RF_name = 'standard_RF_max_depth_10'
save_RF_dis_mat = False

load_RF = True
load_RF_name = 'simple___2021_11_27___22_09_00___standard_RF_max_depth_10'


# In[4]:


assert not (save_RF and load_RF), '"save" and "load" cant both be "True"'


# ## Loading data

# In[5]:


# wavelength grid limits
start_i = 4200
end_i = 12000
    
# load data
print('Loading data and creating dataset')
X = from_s3_npy(s3_client, bucket_name, os.path.join(s3_data_ver_dir_path, 'spec.npy'))
wl_grid = from_s3_npy(s3_client, bucket_name, os.path.join(s3_data_dir_path, 'wl_grid.npy'))
original_wl_str = 'Original wavelength grid: {0}-{1} [A], length={2}.'.format(min(wl_grid),max(wl_grid),len(wl_grid))
print(original_wl_str)

# get the limits of every sample
X_valid = ~np.isnan(X)
sample_i_start = np.argmax(X_valid, axis=1)
sample_i_end = X.shape[1] - np.argmax(np.fliplr(X_valid), axis=1) # non inclusive
support = X_valid.sum(axis=0)/X_valid.shape[0]

# make sure no holes
assert all([i<=2 for i in np.sum(np.abs(np.diff(X_valid, axis=1)), axis=1)]), 'some rows in X contain fragmented spans!'

# limit indices (taking only a slice of the data with full support)
new_wl_str = 'New wavelength grid: {0}-{1} [A], length={2}.'.format(wl_grid[start_i],wl_grid[end_i],end_i-start_i)
print(new_wl_str)

# plot the support of the dataset
fig,ax = plt.subplots(figsize=(12,8))
ax.plot(wl_grid, support)
plt.grid()
plt.axvline(x=wl_grid[start_i], ymin=0, ymax=1, c='r')
plt.axvline(x=wl_grid[end_i], ymin=0, ymax=1, c='r')
ax.set_xlabel('wavelength [A]')
ax.set_ylabel('support %')
ax.set_title('Dataset support')

# taking only the samples with full support over the slice
I_slice = np.array([i for i in range(X_valid.shape[0]) if (sample_i_start[i]<=start_i and sample_i_end[i]>end_i)])
samples_str = 'Number of samples after filtering the slice: {0}'.format(len(I_slice))
print(samples_str)
X = X[I_slice,start_i:end_i]
wl_grid = wl_grid[start_i:end_i]
assert np.sum(np.isnan(X))==0, 'There are still NaN left after filtering.'


# ## Creating train and test sets for RF

# In[6]:


# creaet synthetic samples
print('Creating synthetic data')
shifts = False
if shifts:
    from CustomRandomForest import return_synthetic_data_shift, fix_nan_shifts
    X_syn = return_synthetic_data_shift(X, 10, 3, seed)
    X_syn = fix_nan_shifts(X_syn,10)
else:
    from CustomRandomForest import return_synthetic_data
    X_syn = return_synthetic_data(X, seed)

# merge the data
print('Merging')
from uRF_SDSS import merge_work_and_synthetic_samples
Z, y = merge_work_and_synthetic_samples(X, X_syn)

# train-test split
from sklearn.model_selection import train_test_split
Z_train, Z_test, y_train, y_test, I_train, I_test = train_test_split(Z, y, np.arange(len(y)), train_size=20000, random_state=seed)


# ## Fit a random forest

# In[7]:


if load_RF:
    
    s3_load_dir_path = os.path.join(s3_saves_dir_path, 'RF', load_RF_name)
    print('loading from folder (S3): {0}'.format(s3_load_dir_path))
    
    I_slice = from_s3_npy(s3_client, bucket_name, os.path.join(s3_load_dir_path, 'I_slice.npy'))
    I_train = from_s3_npy(s3_client, bucket_name, os.path.join(s3_load_dir_path, 'I_train.npy'))
    wl_grid = from_s3_npy(s3_client, bucket_name, os.path.join(s3_load_dir_path, 'wl_grid.npy'))
    
    from CustomRandomForest import CustomRandomForest
    rf = CustomRandomForest.load_s3(s3_client, bucket_name, os.path.join(s3_load_dir_path, 'crf.pkl'))
    
else:

    # RF parameters
    N_trees = 500
    min_span = len(wl_grid)
    max_span = len(wl_grid)
    min_samples_split = 1000
    max_features = 'sqrt'
    max_samples = 1.0
    max_depth = 10
    N_snr_bins = 1

    # create a random forest
    from CustomRandomForest import CustomRandomForest
    rf = CustomRandomForest(N_trees=N_trees,
                            min_span=min_span,
                            max_span=max_span,
                            min_samples_split=min_samples_split,
                            max_features=max_features,
                            max_samples=max_samples,
                            max_depth=max_depth
                           )

    # fit the forest to the data
    rf.fit(Z_train, y_train)

    if save_RF:

        # create a save dir
        from datetime import datetime
        s3_save_dir_path = os.path.join(s3_saves_dir_path, 'RF', 'simple___' + datetime.now().strftime("%Y_%m_%d___%H_%M_%S") + '___' + save_RF_name)
        print('save folder (S3): ' + s3_save_dir_path)

        # save some data
        print('Saving numpy arrays')
        to_s3_npy(I_slice, s3_client, bucket_name, os.path.join(s3_save_dir_path, 'I_slice.npy'))
        to_s3_npy(I_train, s3_client, bucket_name, os.path.join(s3_save_dir_path, 'I_train.npy'))
        to_s3_npy(wl_grid, s3_client, bucket_name, os.path.join(s3_save_dir_path, 'wl_grid.npy'))

        # save the random forest
        print('Saving the random forest')
        rf.save_s3(s3_client, bucket_name, os.path.join(s3_save_dir_path, 'crf.pkl'))


# ## Evaluate the RF

# In[8]:


if not load_RF:

    print('Predict on training set')
    y_hat_train = rf.predict(Z_train)

    print('Predict on test set')
    y_hat_test = rf.predict(Z_test)

    print('Evaluating')
    from sklearn.metrics import classification_report
    train_set_report = classification_report(y_train, y_hat_train)
    test_set_report = classification_report(y_test, y_hat_test)
    print('TRAININ-SET:')
    print(train_set_report )
    print('TEST-SET:')
    print(test_set_report)

    if save_RF:



        from CustomRandomForest import log_s3
        log_s3(s3_client, bucket_name, path=os.path.join(s3_save_dir_path, 'RF_log.txt'),
            data = '\n'.join([original_wl_str, new_wl_str, samples_str]),
            N_RF_train_real = sum(y_train==1),
            N_RF_train_syn = len(y_train)-sum(y_train==1),
            N_RF_test_real = sum(y_test==1),
            N_RF_test_syn = len(y_test)-sum(y_test==1),
            N_trees = rf.N_trees,
            min_span = rf.min_span,
            max_span = rf.max_span,
            min_samples_split = rf.min_samples_split,
            max_features = rf.max_features,
            max_samples = rf.max_samples,
            max_depth = rf.max_depth,
            train_set_report = '\n'+train_set_report,
            test_set_report = '\n'+test_set_report
            )


# ###  Plots

# In[9]:


if not load_RF:

    p_test = rf.predict_proba(Z_test)
    p_train = rf.predict_proba(Z_train)

    fig = plt.figure()
    plt.hist(p_train[y_train==1,1], density=True, bins=20, alpha=0.5, label='train - real')
    plt.hist(p_train[y_train==2,2], density=True, bins=20, alpha=0.5, label='train - synthetic')
    plt.hist(p_test[y_test==1,1], density=True, bins=20, alpha=0.5, label='test - real')
    plt.hist(p_test[y_test==2,2], density=True, bins=20, alpha=0.5, label='test - synthetic')
    plt.legend()
    plt.title("probability distribution soft predictions")
    plt.ylabel("Pr")
    plt.xlabel("predicted probability")

    if save_RF:
        to_s3_fig(fig, s3_client, bucket_name, os.path.join(s3_save_dir_path, 'prob_dist.png'))


# ## Calculate similarity matrix, weirdness scores and T-SNE

# In[10]:


if load_RF:
    
    print('loading...')
    X = from_s3_npy(s3_client, bucket_name, os.path.join(s3_load_dir_path, 'X.npy'))
    X_leaves = from_s3_npy(s3_client, bucket_name, os.path.join(s3_load_dir_path, 'X_leaves.npy'))
    Y_hat = from_s3_npy(s3_client, bucket_name, os.path.join(s3_load_dir_path, 'Y_hat.npy'))
    sim_mat = from_s3_npy(s3_client, bucket_name, os.path.join(s3_load_dir_path, 'sim_mat.npy'))
    
else:

    # Throwing the RF's test set, and the train synthetic spectra
    """
    because merge_work_and_synthetic_samples concatenates the N synthetic spectra after the N real spectra,
    all the train indices up to N are real.
    """
    X = X[I_train[I_train<X.shape[0]]]

    print('Applying the RF on the full dataset (real spectra only)')
    X_leaves = rf.apply(X)

    print('Predicting fully')
    Y_hat = rf.predict_full_from_leaves(X_leaves)

    print('Calculating the similarity matrix')
    from CustomRandomForest import build_similarity_matrix
    sim_mat = build_similarity_matrix(X_leaves, Y_hat)

    if save_RF:

        print('Saving the data')
        to_s3_npy(X, s3_client, bucket_name, os.path.join(s3_save_dir_path, 'X.npy'))
        to_s3_npy(X_leaves, s3_client, bucket_name, os.path.join(s3_save_dir_path, 'X_leaves.npy'))
        to_s3_npy(Y_hat, s3_client, bucket_name, os.path.join(s3_save_dir_path, 'Y_hat.npy'))

        print('Saving the similarity matrix')
        to_s3_npy(sim_mat, s3_client, bucket_name, os.path.join(s3_save_dir_path, 'sim_mat.npy'))


# In[11]:


print('Calculating the weirdness scores')
dis_mat = 1 - sim_mat
weird_scores = np.mean(dis_mat, axis=1)

if save_RF:

    print('Saving the weirdness scores')
    to_s3_npy(weird_scores, s3_client, bucket_name, os.path.join(s3_save_dir_path, 'weird_scores.npy'))

    if save_RF_dis_mat:
        print('Saving the dissimilarity matrix')
        to_s3_npy(dis_mat, s3_client, bucket_name, os.path.join(s3_save_dir_path, 'dis_mat.npy'))


# In[12]:


if load_RF:
    
    print('loading...')
    sne = from_s3_npy(s3_client, bucket_name, os.path.join(s3_load_dir_path, 'tsne.npy'))
    
else:

    print('Running T-SNE')
    from sklearn.manifold import TSNE
    sne = TSNE(n_components=2, perplexity=25, metric='precomputed', verbose=1, random_state=seed).fit_transform(dis_mat)

    if save_RF:

        print('Saving T-SNE')
        to_s3_npy(sne, s3_client, bucket_name, os.path.join(s3_save_dir_path, 'tsne.npy'))


# # Plots

# In[13]:


if not load_RF:
    
    fig = plt.figure()
    tmp = plt.hist(weird_scores, bins=60, color="g")
    plt.title("Weirdness score histogram")
    plt.ylabel("N")
    plt.xlabel("weirdness score")

    if save_RF:
        to_s3_fig(fig, s3_client, bucket_name, os.path.join(s3_save_dir_path, 'weirdness_scores_histogram.png'))


# In[14]:


if not load_RF:

    distances = dis_mat[np.tril_indices(dis_mat.shape[0], -1)]

    fig = plt.figure()
    tmp = plt.hist(distances, bins=100)
    plt.title("Distances histogram")
    plt.ylabel("N")
    plt.xlabel("distance")

    if save_RF:
        to_s3_fig(fig, s3_client, bucket_name, os.path.join(s3_save_dir_path, 'distances_histogram.png'))


# In[15]:


if not load_RF:
    
    sne_f1 = sne[:, 0]
    sne_f2 = sne[:, 1]

    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111)
    im_scat = ax.scatter(sne_f1, sne_f2, s=3, c=weird_scores, cmap=plt.cm.get_cmap('jet'), picker=1)
    ax.set_xlabel('t-SNE Feature 1')
    ax.set_ylabel('t-SNE Feature 2')
    ax.set_title(r't-SNE Scatter Plot Colored by Weirdness score')
    clb = fig.colorbar(im_scat, ax=ax)
    clb.ax.set_ylabel('Weirdness', rotation=270)
    plt.show()

    if save_RF:
        to_s3_fig(fig, s3_client, bucket_name, os.path.join(s3_save_dir_path, 'tsne_colored_by_weirdness.png'))


# In[16]:


gs = from_s3_pkl(s3_client, bucket_name, os.path.join(s3_data_ver_dir_path, 'gs.pkl'))
I_real_train = I_train[I_train<len(I_slice)]
snr = gs.snMedian.iloc[I_slice[I_real_train]]


# In[17]:


if not load_RF:
    
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111)
    import matplotlib.colors as colors
    im_scat = ax.scatter(sne_f1, sne_f2, s=3, c=snr, cmap=plt.cm.get_cmap('jet'), norm=colors.LogNorm(vmin=snr.min(), vmax=80))
    ax.set_xlabel('t-SNE Feature 1')
    ax.set_ylabel('t-SNE Feature 2')
    ax.set_title(r't-SNE Scatter Plot Colored by SNR')
    clb = fig.colorbar(im_scat, ax=ax)
    clb.ax.set_ylabel('SNR', rotation=270)
    plt.show()

if save_RF:
    to_s3_fig(fig, s3_client, bucket_name, os.path.join(s3_save_dir_path, 'tsne_colored_by_snr.png'))


# ## calculating decisions path

# if save_RF:
#     
#     # Creating the decision paths dictionary
#     from progressbar import progressbar
#     decision_paths = dict()
#     for k in progressbar(range(len(rf.estimators_))):
#         i_real = np.where(Y_hat[:,k]==1)[0] # indices of all samples which the k-th tree classified as "real"
#         i_sample,i_node = rf.estimators_[k].tree_.decision_path(X[i_real,rf.tree_i_start[k]:rf.tree_i_end[k]].astype(np.float32)).nonzero()
#         i_feature = rf.tree_i_start[k]+rf.estimators_[k].tree_.feature[i_node]
#         for i in range(len(i_real)):
#             temp_feature = i_feature[i_sample==i]
#             decision_paths[(i_real[i],k)] = temp_feature[:-1] # the last node is a leaf (not a decision node)
# 
#     # saving it
#     with open(os.path.join(save_dir_path, 'decision_paths.pkl'), 'wb') as f:
#         pickle.dump(decision_paths, f)

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
    load_NN_dir_path = os.path.join(saves_dir_path, 'NN', load_NN_name)
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


if not load_NN:

    from NN import DistillationDataGenerator

    X_train, X_val, I_train, I_test = train_test_split(X, np.arange(X.shape[0]), train_size=9000, random_state=seed)

    batch_size = 128
    
    full_epoch = int(sys.argv[2])
    if len(sys.argv)>3:
        snr_range_db = [int(sys.argv[3]),int(sys.argv[4])]
    else:
        snr_range_db = None

    from NN import DistillationDataGenerator
    train_gen_full = DistillationDataGenerator(X_train, dis_mat[I_train,:][:,I_train], batch_size=batch_size, shuffle=True, seed=seed, snr_range_db=snr_range_db, full_epoch=full_epoch)
    train_gen = DistillationDataGenerator(X_train, dis_mat[I_train,:][:,I_train], batch_size=batch_size, shuffle=True, seed=seed, snr_range_db=snr_range_db, full_epoch=False)
    val_gen = DistillationDataGenerator(X_val, dis_mat[I_test,:][:,I_test], batch_size=batch_size, shuffle=True, seed=seed, snr_range_db=snr_range_db, full_epoch=False)


# ## Training the model

# In[25]:


if save_NN:
    
    # create a save dir
    from datetime import datetime
    run_prefix = os.environ['RUN_PREFIX']
    s3_save_NN_dir_path = os.path.join(s3_saves_dir_path, 'NN', run_prefix+'___' + datetime.now().strftime("%Y_%m_%d___%H_%M_%S") + '___' + save_NN_name)
    print('save NN folder (S3): ' + s3_save_NN_dir_path)


# In[26]:


if not load_NN:

    epochs = 5 # [full]
    sub_epochs = 1 # [full]
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
            history = siamese_model.fit(train_gen_full, epochs=sub_epochs, validation_data=val_gen)
        except KeyError:
            history = siamese_model.fit(train_gen_full, epochs=sub_epochs, validation_data=val_gen)
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
    
    # the path of the RF this NN was trained on
    s3_save_RF_path = 'RF not save'
    if save_RF:
        s3_save_RF_path = s3_save_dir_path
    if load_RF:
        s3_save_RF_path = s3_load_dir_path

    # save log
    from s3 import log_s3
    log_s3(s3_client, bucket_name, s3_save_NN_dir_path, 'NN_log.txt',
        s3_save_RF_path = s3_save_RF_path,
        embedding_summary = embedding_summary,
        siamese_network_summary = siamese_network_summary
        )
    
    # save the network
    s3_model_path = os.path.join(s3_save_NN_dir_path, 'model')
    from s3 import s3_save_TF_model
    s3_save_TF_model(siamese_model, s3_client, bucket_name, s3_model_path)

