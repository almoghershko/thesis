import io
import os
import numpy as np
import pickle
from urllib.parse import urlparse
import boto3
import zipfile
import tempfile
import tensorflow as tf

# STORE

def to_s3_npy(data: np.array, s3_client, bucket_name, path_in_bucket: str):
    s3_uri = f"s3://{bucket_name}/{path_in_bucket}"
    print('saving to uri: '+s3_uri)
    bytes_ = io.BytesIO()
    np.save(bytes_, data, allow_pickle=True)
    bytes_.seek(0)
    parsed_s3 = urlparse(s3_uri)
    s3_client.upload_fileobj(
        Fileobj=bytes_, Bucket=parsed_s3.netloc, Key=parsed_s3.path[1:]
    )
    return True

def to_s3_pkl(data, s3_client, bucket_name, path_in_bucket: str):
    s3_uri = f"s3://{bucket_name}/{path_in_bucket}"
    print('saving to uri: '+s3_uri)
    data_io = io.BytesIO()
    pickle.dump(data, data_io)
    data_io.seek(0)
    s3_client.upload_fileobj(data_io, bucket_name, path_in_bucket)
    return True

def to_s3_fig(fig, s3_client, bucket_name, path_in_bucket: str):
    s3_uri = f"s3://{bucket_name}/{path_in_bucket}"
    print('saving to uri: '+s3_uri)
    data_io = io.BytesIO()
    fig.savefig(data_io)
    data_io.seek(0)
    s3_client.upload_fileobj(data_io, bucket_name, path_in_bucket)
    return True

def zipdir(path, ziph):
    # Zipfile hook to zip up model folders
    length = len(path)
    for root, dirs, files in os.walk(path):
        folder = root[length:] # Stop zipping parent folders
        for file in files:
            ziph.write(os.path.join(root, file), os.path.join(folder, file))

def s3_save_TF_model(model, s3_client, bucket_name, path_in_bucket, model_name='model'):
    with tempfile.TemporaryDirectory() as tempdir:
        # save
        model.save(f"{tempdir}/{model_name}")
        # zip
        zipf = zipfile.ZipFile(f"{tempdir}/{model_name}.zip", "w", zipfile.ZIP_STORED)
        zipdir(f"{tempdir}/{model_name}", zipf)
        zipf.close()
        # upload
        s3_uri = f"s3://{bucket_name}/{path_in_bucket}"
        print('saving to uri: '+s3_uri)
        s3_client.upload_file(
            f"{tempdir}/{model_name}.zip",
            bucket_name,
            path_in_bucket
        )
        
def log_s3(s3_client, bucket_name, path_in_bucket, log_name, **kwargs):
    s3_uri = f"s3://{bucket_name}/{path_in_bucket}/{log_name}"
    print('saving to uri: '+s3_uri)
    with tempfile.TemporaryDirectory() as tempdir:
        with open(f"{tempdir}/{log_name}", 'w') as f:
            for key,value in kwargs.items():
                f.write('{0} = {1}\n'.format(key,value))
        s3_client.upload_file(
            f"{tempdir}/{log_name}",
            bucket_name,
            f"{path_in_bucket}/{log_name}"
        )

# LOAD

def from_s3_npy(s3_client, bucket_name, path_in_bucket: str):
    s3_uri = f"s3://{bucket_name}/{path_in_bucket}"
    print('loading from uri: '+s3_uri)
    bytes_ = io.BytesIO()
    parsed_s3 = urlparse(s3_uri)
    s3_client.download_fileobj(
        Fileobj=bytes_, Bucket=parsed_s3.netloc, Key=parsed_s3.path[1:]
    )
    bytes_.seek(0)
    return np.load(bytes_, allow_pickle=True)

def from_s3_pkl(s3_client, bucket_name, path_in_bucket: str):
    s3_uri = f"s3://{bucket_name}/{path_in_bucket}"
    print('loading from uri: '+s3_uri)
    data_io = io.BytesIO()
    s3_client.download_fileobj(bucket_name, path_in_bucket, data_io)
    data_io.seek(0)
    data = pickle.load(data_io)
    return data

def s3_load_TF_model(s3_client, bucket_name, path_in_bucket, model_name='model', custom_objects={}):
    with tempfile.TemporaryDirectory() as tempdir:
        # Fetch and save the zip file to the temporary directory
        s3_client.download_file(
            bucket_name,
            path_in_bucket,
            f"{tempdir}/{model_name}.zip"
        )
        # Extract the model zip file within the temporary directory
        with zipfile.ZipFile(f"{tempdir}/{model_name}.zip") as zip_ref:
            zip_ref.extractall(f"{tempdir}/{model_name}")
        # Load the keras model from the temporary directory
        return tf.keras.models.load_model(f"{tempdir}/{model_name}", custom_objects=custom_objects)