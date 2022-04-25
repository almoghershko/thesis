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
s3_data_ver_dir_path = os.path.join(s3_data_dir_path,'100K_V2')
s3_runs_dir_path = os.path.join(s3_work_dir_path , 'runs')

s3_client = boto3.client("s3", endpoint_url=endpoint_url)

# adding code folder to path
sys.path.insert(1, local_code_dir_path)
from s3 import to_s3_npy, to_s3_pkl, from_s3_npy, from_s3_pkl, to_s3_fig

# get run directory
run_name = sys.argv[1]
s3_run_dir_path = os.path.join(s3_runs_dir_path, run_name)
print('run dir path = {0}'.format(s3_run_dir_path))


# get the chunk number and chunk
i_gs = sys.argv[2]
print('chunk index = {0}'.format(i_gs))
gs = from_s3_pkl(s3_client, bucket_name, os.path.join(s3_data_ver_dir_path,'gs{0}.pkl'.format(i_gs)))

# create a wrapper that returns the index also
from pre_processing import download_spectrum
def download_spectrum_wrapper(i):
    spec, _, ivar = download_spectrum(gs.iloc[i], pre_proc=True, wl_grid=wl_grid)
    return i, spec, ivar

# create jobs to download and preprocess
from joblib import Parallel, delayed
res = Parallel(n_jobs=-1, verbose=5, prefer="threads")(delayed(download_spectrum_wrapper)(i) for i in range(len(gs)))

# fiter the good results only (exception encountered during download will return empty arrays)
res = sorted(res, key=lambda x: x[0])
goodRes = [len(val[1]) > 0 for val in res]
gs = gs[goodRes]
gs.index = range(len(gs))
from itertools import compress
res = list(compress(res, goodRes))

# save the spectra
spec = np.stack([x[1] for x in res], axis=0)
if not len([i for i in range(spec.shape[0]) if np.any(np.isnan(spec[i]))])==0: # making sure no NaNs
    print("<<<<< ERROR >>>>>>: NaN foudn in dataset!")
to_s3_npy(spec, s3_client, bucket_name, os.path.join(s3_data_ver_dir_path,'spec{0}.npy'.format(i_gs)))

# save the ivar
ivar = np.stack([x[2] for x in res], axis=0)
to_s3_npy(ivar, s3_client, bucket_name, os.path.join(s3_data_ver_dir_path,'ivar{0}.npy'.format(i_gs)))