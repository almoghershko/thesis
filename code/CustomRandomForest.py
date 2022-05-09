# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 14:43:14 2020

@author: AlmogHershko
"""

from sklearn.tree import DecisionTreeClassifier
import numpy as np
from joblib import Parallel, delayed
from progressbar import progressbar
import pickle

MAX_SEED = 2**32 - 1

# importing some L/S functions from S3 module
import os
import sys
local_home_dir_path = os.path.expanduser("~")
local_work_dir_path = os.path.join(local_home_dir_path, 'workdir3')
local_code_dir_path = os.path.join(local_work_dir_path , 'code')
sys.path.insert(1, local_code_dir_path)
from s3 import to_s3_npy, to_s3_pkl, from_s3_npy, from_s3_pkl

class CustomRandomForest:
    def __init__(self, N_trees, min_span, max_span, min_samples_split=2, max_features='sqrt', max_samples=1.0, max_depth=None, N_snr_bins=1, snr=None, seed=42):
        """
        Initialize the random forest.

        Parameters
        ----------
        N_trees : int
            Number of trees.
        min_span : int
            the minimal span of features to train trees on.
        max_span : int
            the maximal span of features to train trees on.
        max_features : int, float or {“auto”, “sqrt”, “log2”}, default='sqrt'
            The number of features to consider when looking for the best split
        min_samples_split : int or float, default=2
            The minimum number of samples required to split an internal node
        max_samples : float between 0 and 1, default 1.0
            The fraction of samples to draw (with replacement) from X to train
            each base estimator.
        max_depth : int, default=None
            The maximum depth of the tree. If None, then nodes are expanded
            until all leaves are pure or until all leaves contain less than
            min_samples_split samples.
        N_snr_bins : int, default=1
            The number of SNR bins to split the data to. If N_snr_bins>1 then snr should be given.
        snr : array
            the snr of each sample.

        Returns
        -------
        None.

        """
        
        # save the params
        self.N_trees = N_trees
        self.min_span = min_span
        self.max_span = max_span
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.max_samples = max_samples
        self.max_depth = max_depth
        self.N_snr_bins = N_snr_bins
        self.snr = snr
        
        # create a random seed for each tree
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.tree_seeds = np.round(MAX_SEED*self.rng.random(self.N_trees)).astype(int)
        
    def fit(self, X, y, prefer="threads", n_jobs=-1):
        """
        Fits the random forest to the data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input samples.
        y : ndarray of shape (n_samples,)
            The classes.

        Returns
        -------
        None.

        """
        
        # verify data is complete
        X_valid = ~np.isnan(X)
        self._validate_X(X_valid)
        
        # save the data
        self.N_samples = X.shape[0]
        self.N_features = X.shape[1]
        self.N_classes = len(set(y))
        
        # get the limits of every sample
        self.sample_i_start = np.argmax(X_valid, axis=1)
        self.sample_i_end = X.shape[1] - np.argmax(np.fliplr(X_valid), axis=1) # non inclusive
        self.sample_spans = self.sample_i_end - self.sample_i_start
        
        del X_valid
        
        # verify max_span is valid
        if not max(self.sample_spans) >= self.max_span:
            print('warning: no object has span greater than the maximal span specified (%d). Overriding max_span to be the minimal span of the objects at hand (%d).' % (self.max_span, min(self.sample_spans)))
            self.max_span = min(self.sample_spans)
        
        # randomly select spans from a uniform distribution
        spans = np.round(self.min_span + (self.max_span-self.min_span) * self.rng.random(self.N_trees)).astype(int)
        
        # create SNR bins
        #tree2bin = {i:np.ceil((i+1)/(self.N_trees/self.N_snr_bins))-1 for i in range(self.N_trees)}
        #snr_arg_sort = np.argsort(self.snr)
        #snr_edges = self.snr[snr_arg_sort[np.round(np.linspace(0, len(self.snr)-1, self.N_snr_bins+1)).astype(int)]]
        #snr_edges[-1] += 1 # to include the last one
        #bin2samples = {i:np.argwhere((self.snr>=snr_edges[i]) & (self.snr<snr_edges[i+1])).reshape(-1) for i in range(self.N_snr_bins)}

        # train all trees in parallel
        #res = Parallel(n_jobs=-1, verbose=5, prefer="threads")(delayed(self._fit_single)(i, spans[i], X[bin2samples[tree2bin[i]]], y[bin2samples[tree2bin[i]]]) for i in range(self.N_trees))
        res = Parallel(n_jobs=n_jobs, verbose=5, prefer=prefer)(delayed(self._fit_single)(i, spans[i], X, y) for i in range(self.N_trees))
        """
        res = []
        for i in progressbar(range(self.N_trees)):
            res.append(self._fit_single(i, spans[i], X, y))
        """
        
        # sort results after the parallel execution
        res = sorted(res, key=lambda x: x[0])
        
        # save results
        self.estimators_ = [x[1] for x in res]
        self.tree_i_start = [x[2] for x in res]
        self.tree_i_end = [x[3] for x in res]
        self.I_oob = [x[4] for x in res]
        self.leaves_pred = [x[5] for x in res]
        
    def _validate_X(self, X_valid):
        """
        Make sure each line in the boolean matrix X_valid have a continuous segment of "True"s

        Parameters
        ----------
        X_valid : 2-D ndarray of type bool
            a boolean matrix of indicators for a valid data place.

        Returns
        -------
        True if valid, False if not.

        """
        
        assert np.min(np.sum(X_valid, axis=1)) > 0, 'some rows in X has no valid samples!'
        assert all([i<=2 for i in np.sum(np.abs(np.diff(X_valid, axis=1)), axis=1)]), 'some rows in X contain fragmented spans!'
        
    def _fit_single(self, i_tree, span, X, y):
        """
        Fits a single tree on a limited part of the features range.

        Parameters
        ----------
        i_tree : int
            the index of the tree.
        span : int
            the span for this.
        X : ndarray of shape (n_samples, n_features)
            The input samples.
        y : ndarray of shape (n_samples,)
            The classes.

        Returns
        -------
        i_tree : int
            the index of the tree.
        estimator : DecisionTreeClassifier object
            the decision tree
        i_start : int
            the index of the first feature
        i_end : int
            1 + the index of the last feature
        I_oob : list of integers
            the indices of samples that were not used to train (out of bag)
        leaves_pred : array of floats of shape (n_leaves,)
            the class predicted for each node (only some of them are leavse, rest are decision nodes)
        """
        
        # creating a random generator
        rng = np.random.default_rng(self.tree_seeds[i_tree])
        
        flag = True
        cnt = 0
        MAX_RUNS = 5
        
        while(flag):
            # create a histogram (each bin counts how many examples contains this span, starting from this bin)
            h = np.zeros(self.N_features, dtype=int)
            for i in range(self.N_samples):
                if self.sample_spans[i] >= span:
                    h[self.sample_i_start[i]] += 1
                    h[self.sample_i_end[i]-span+1] -= 1
            h = np.cumsum(h)

            # sample from this histogram
            population = np.nonzero(h)[0]
            weights = h[population]
            i_start = rng.choice(population, size=1, replace=True, p=weights/sum(weights))[0]
            i_end = i_start + span

            del h, population, weights

            # extract data for training
            X_i = X[:,i_start:i_end]
            valid = (~np.isnan(X_i)).all(axis=1)
            X_i = X_i[valid,:]
            y_i = y[valid]
            N_i = len(y_i)

            del valid

            # bootstrap - choosing samples randomly
            I = rng.choice(np.arange(N_i), size=int(np.ceil(self.max_samples*N_i)))
            I_oob = set(np.arange(N_i)).difference(set(I))
            X_i = X_i[I]
            y_i = y_i[I]

            # train
            estimator = DecisionTreeClassifier(max_features=self.max_features,
                                               min_samples_split=self.min_samples_split,
                                               max_depth=self.max_depth,
                                               random_state=round(MAX_SEED*rng.random(1)[0])
                                              )
            estimator = estimator.fit(X_i, y_i)
            cnt += 1
            
            # making sure tree has splits (in some cases it may not split, like if the span chosen too few of samples - less then min_samples_split)
            if (estimator.tree_.max_depth>0):
                flag = False
            else:
                if cnt==MAX_RUNS:
                    raise Exception("Had %d chances to fit this tree, and still no split..." % (MAX_RUNS))
        
        # create leaf-prediction mapping
        leaves_pred = np.squeeze(estimator.classes_[np.argmax(estimator.tree_.value, axis=2)])
        
        return i_tree, estimator, i_start, i_end, I_oob, leaves_pred
    
    def apply(self, X):
        """
        Apply trees in the forest to X, return leaf indices.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        X_leaves : ndarray of shape (n_samples, n_estimators)
            For each datapoint x in X and for each tree in the forest,
            return the index of the leaf x ends up in.

        """

        # verify data is complete
        X_valid = ~np.isnan(X)
        self._validate_X(X_valid)

        # get the limits of every sample
        sample_i_start = np.argmax(X_valid, axis=1)
        sample_i_end = X.shape[1] - np.argmax(np.fliplr(X_valid), axis=1) # non inclusive
        
        # predict with all trees in parallel
        print("apply: starting %d jobs" % self.N_trees)
        res = Parallel(n_jobs=-1, verbose=5, prefer="threads")(delayed(apply_single_tree)(i_tree, self.estimators_[i_tree], X, sample_i_start, sample_i_end, self.tree_i_start[i_tree], self.tree_i_end[i_tree]) for i_tree in range(self.N_trees))
        
        # sort results
        res = sorted(res, key=lambda x: x[0])
        X_leaves = np.array([val[1] for val in res])
        
        return X_leaves.T
    
    def predict_full_from_leaves(self, X_leaves):
        """
        Like predict_full, but from the leaves matrix (after calling apply) insted of the data.
        Predict class for X from each tree.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        Y_hat : ndarray of shape (n_samples, n_trees)
            The predicted classes.

        """
        
        # predict with all trees in parallel
        res = Parallel(n_jobs=-1, verbose=5, prefer="threads")(delayed(predict_from_leaves_i)(i_tree, X_leaves[:,i_tree], self.leaves_pred[i_tree]) for i_tree in range(self.N_trees))

        # sort results
        res = sorted(res, key=lambda x: x[0])
        Y_hat = np.array([val[1] for val in res])
        
        return Y_hat.T
    
    def predict_from_leaves(self, X_leaves):
        """
        Like predict, but from the leaves matrix (after calling apply) insted of the data.
        Predict class for X

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y_hat : ndarray of shape (n_samples,)
            The predicted classes.

        """
        
        Y_hat = self.predict_full_from_leaves(X_leaves)
        y_hat = self.predict_from_full(Y_hat)
        return y_hat
    
    def predict_from_full(self, Y_hat):
        """
        Like predict, but from the full predictions matrix (after calling predict_full) insted of the data.
        Predict class for X

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y_hat : ndarray of shape (n_samples,)
            The predicted classes.

        """
        
        y_hat = np.array([np.argmax(np.bincount(Y_hat[i,~np.isnan(Y_hat[i])].astype(int), minlength=self.N_classes+1)) for i in range(len(Y_hat))])
        
        return y_hat
    
    def predict(self, X):
        """
        Predict class for X

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y_hat : ndarray of shape (n_samples,)
            The predicted classes.

        """
        
        X_leaves = self.apply(X)
        y_hat = self.predict_from_leaves(X_leaves)
        return y_hat
    
    def predict_full(self, X):
        """
        Predict class for X from each tree.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        Y_hat : ndarray of shape (n_samples, n_trees)
            The predicted classes.

        """
        
        X_leaves = self.apply(X)
        Y_hat = self.predict_full_from_leaves(X_leaves)
        return Y_hat
    
    def save(self, file_path):
        """
        pickles everything needed to a file so the random forest could be loaded again.

        Parameters
        ----------
        file_path : string
            path to the pickle file.

        Returns
        -------
        None.

        """
        
        # create a dictionary of all the attributes which are not methods
        params = {attr_name: getattr(self, attr_name) for attr_name in dir(self) if ((not hasattr(getattr(self, attr_name), '__call__')) and (attr_name[0]!='_'))}
        
        # saving the attributes dictionary
        with open(file_path,'wb') as f:
            pickle.dump(params, f, protocol=pickle.HIGHEST_PROTOCOL)
            
    def save_s3(self, s3_client, bucket_name, path_in_bucket: str):
        """
        pickles everything needed to a file in S3 so the random forest could be loaded again.

        Parameters
        ----------
        s3_client : botocore.client.S3
            the S3 boto client
        bucket_name : string
            the name of the bucket
        path_in_bucket : string
            path of the pickle file relative to the bucket (path should start at the prefix)

        Returns
        -------
        None.

        """
        
        # create a dictionary of all the attributes which are not methods
        params = {attr_name: getattr(self, attr_name) for attr_name in dir(self) if ((not hasattr(getattr(self, attr_name), '__call__')) and (attr_name[0]!='_'))}
        
        # saving the attributes dictionary to S3
        to_s3_pkl(params, s3_client, bucket_name, path_in_bucket)
            
    def predict_proba_from_full(self, Y_hat):
        """
        Like predict_proba, but from a full prediction matrix.

        Parameters
        ----------
        Y_hat : ndarray of shape (n_samples, n_trees)
            The predicted classes.

        Returns
        -------
        p : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples

        """
        
        p = np.array([np.bincount(Y_hat[i,~np.isnan(Y_hat[i])].astype(int), minlength=self.N_classes+1) for i in range(len(Y_hat))])
        row_sums = p.sum(axis=1)
        p = p / row_sums[:, np.newaxis]
        
        return p
    
    def predict_proba(self, X):
        """
        Predict class probabilities for X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        p : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples

        """
        
        Y_hat = self.predict_full(X)
        p = self.predict_proba_from_full(Y_hat)
        
        return p
            
    def load(file_path):
        """
        load a save file of a custom random forest and returns a custom random forest object.

        Parameters
        ----------
        file_path : string
            path to the pickle file.

        Returns
        -------
        a CustomRandomForest object.

        """
        
        # loading the param dict
        with open(file_path,'rb') as f:
            params = pickle.load(f)
        
        # creating a custom random and restoring the attributes
        rf = CustomRandomForest(params['N_trees'], ['min_span'], ['max_span'])
        for attr_name, attr in params.items():
            setattr(rf, attr_name, attr)
        
        return rf
    
    def load_s3(s3_client, bucket_name, path_in_bucket: str):
        """
        load a save file from S3 of a custom random forest and returns a custom random forest object.

        Parameters
        ----------
        s3_client : botocore.client.S3
            the S3 boto client
        bucket_name : string
            the name of the bucket
        path_in_bucket : string
            path of the pickle file relative to the bucket (path should start at the prefix)

        Returns
        -------
        a CustomRandomForest object.

        """
        
        # loading the param dict from S3
        params = from_s3_pkl(s3_client, bucket_name, path_in_bucket)
        
        # creating a custom random and restoring the attributes
        rf = CustomRandomForest(params['N_trees'], ['min_span'], ['max_span'])
        for attr_name, attr in params.items():
            setattr(rf, attr_name, attr)
        
        return rf
    
def apply_single_tree(i_tree, tree, X, sample_i_start, sample_i_end, tree_i_start, tree_i_end):
    """
    apply a single tree on the samples.

    Parameters
    ----------
    i_tree : int
        the index of the tree in the forest.
    tree : DecisionTreeClassifier
        the tree object.
    X : ndarray of shape (n_samples, n_features)
        the data matrix.
    sample_i_start : ndarray of shape (n_samples,)
        the index of the first valid feature of each sample.
    sample_i_end : ndarray of shape (n_samples,)
        one plut the index of the last valid feature of each sample.
    tree_i_start : int
        the index of the first feature in the span of the tree.
    tree_i_end : int
        one plus the index of the last feature in the span of the tree.

    Returns
    -------
    i_tree : int
        the index of the tree in the forest.
    x_leaves : ndarray of shape (n_samples)
        the leaf index each sample ended up in the tree.

    """
    
    x_leaves = np.zeros(shape=(X.shape[0],))

    # determin which samples can this tree be applied to, and apply to them. rest will get NaN.
    valid = np.array([(sample_i_start[i]<=tree_i_start) and (sample_i_end[i]>=tree_i_end) for i in range(X.shape[0])], dtype=bool)
    x_leaves[valid] = tree.apply(X[valid,tree_i_start:tree_i_end])
    x_leaves[~valid] = np.nan
        
    return i_tree, x_leaves

def predict_from_leaves_i(i_tree, x_leaves, leaves_pred):
    """
    convert leaves indices to predictions for a single tree.

    Parameters
    ----------
    i_tree : int
        the index of the tree in the forest.
    x_leaves : ndarray of shape (n_samples,)
        the leaf index of each sample, for this tree.
    leaves_pred : array of floats of shape (n_leaves,)
        the class predicted for each node (only some of them are leavse, rest are decision nodes), for this tree

    Returns
    -------
    i_tree : int
        the index of the tree in the forest.
    x_pred : ndarray of shape (n_samples,)
        the prediction for each sample, for this tree.

    """
    
    x_pred = np.zeros(shape=x_leaves.shape)
    x_pred[np.isnan(x_leaves)] = np.nan
    x_pred[~np.isnan(x_leaves)] = leaves_pred[x_leaves[~np.isnan(x_leaves)].astype(int)]
    
    return i_tree, x_pred
    
def build_similarity_matrix(X_leaves, Y_hat):
    """
    calculate the similarity matrix between objects after a custom random forest
    has been trained to distinguish between real and synthetic data.
    The similarity between object i and j is defined by the number of times the
    objects ended up on the same terminal node in a tree that classified them
    both correctly as real, divided by the number of trees classified them both
    correctly as real.

    Parameters
    ----------
    X_leaves : 2-D numpy ndarray of shape (n_samples, n_trees)
        the leaves matrix received by applying the random forest.
    Y_hat : 2-D numpy ndarray of shape (n_samples, n_trees)
        the predictions of each tree for each sample.

    Returns
    -------
    sim_mat : 2-D numpy ndarray of shape (n_samples, n_samples)
        the similarity matrix.

    """

    N_samples = X_leaves.shape[0]
    good_preds = Y_hat == 1
    res = Parallel(n_jobs=-1, verbose=5)(delayed(sim_i)(i, X_leaves, good_preds) for i in range(N_samples))
    print('assembling the matrix.')
    res = sorted(res, key=lambda x: x[0])
    sim_mat = np.array([x[1] for x in res])
    i_lower = np.tril_indices(N_samples, -1)
    sim_mat[i_lower] = sim_mat.T[i_lower]  # make the matrix symmetric
    
    return sim_mat

def sim_i(i, X_leaves, good_preds):
    """
    calculate the similarities of a single line in the similarity matrix.
    ignoring the diagonal and lower triangular part of the matrix (returned 0 in these places).

    Parameters
    ----------
    i : int
        line index.
    X_leaves : ndarray of shape (n_samples, n_trees)
        the leaves matrix (after applying the random forest).
    good_preds : ndarray of shape (n_samples, n_trees) and type bool
        a boolean matrix where the value in location (i,j) is True if the j-th
        tree predicted the i-th sample as real, False of not.

    Returns
    -------
    i : int
        line index.
    sim : ndarray of shape (n_samples,)
        the i-th line in the similarity matrix (only right to the main diagonal).

    """
    
    N_samples = X_leaves.shape[0]
    sim = np.zeros(shape=(N_samples,), dtype=np.float32)
    sim[i] = 1.0 # trivial
    for j in range(i+1, N_samples):
        valid = np.where(np.logical_and(good_preds[i,:], good_preds[j,:]))[0]
        if len(valid)>0:
            same = X_leaves[i,valid] == X_leaves[j,valid]
            sim[j] = np.sum(same) / len(valid)
    return (i, sim)

def build_distance_matrix(X_leaves, Y_hat):
    """
    distance=1-similarity

    Parameters
    ----------
    X_leaves : 2-D numpy ndarray of shape (n_samples, n_trees)
        the leaves matrix received by applying the random forest.
    Y_hat : 2-D numpy ndarray of shape (n_samples, n_trees)
        the predictions of each tree for each sample.

    Returns
    -------
    dist_mat : 2-D numpy ndarray of shape (n_samples, n_samples)
        the distance matrix.

    """

    N_samples = X_leaves.shape[0]
    good_preds = Y_hat == 1
    res = Parallel(n_jobs=-1, verbose=5)(delayed(dist_i)(i, X_leaves, good_preds) for i in range(N_samples))
    print('assembling the matrix.')
    res = sorted(res, key=lambda x: x[0])
    dist_mat = np.array([x[1] for x in res])
    i_lower = np.tril_indices(N_samples, -1)
    dist_mat[i_lower] = dist_mat.T[i_lower]  # make the matrix symmetric
    
    return dist_mat

def dist_i(i, X_leaves, good_preds):
    """
    distance=1-similarity

    Parameters
    ----------
    i : int
        line index.
    X_leaves : ndarray of shape (n_samples, n_trees)
        the leaves matrix (after applying the random forest).
    good_preds : ndarray of shape (n_samples, n_trees) and type bool
        a boolean matrix where the value in location (i,j) is True if the j-th
        tree predicted the i-th sample as real, False of not.

    Returns
    -------
    i : int
        line index.
    dist : ndarray of shape (n_samples,)
        the i-th line in the distance matrix (only right to the main diagonal).

    """
    
    N_samples = X_leaves.shape[0]
    dist = np.zeros(shape=(N_samples,), dtype=np.float32)
    dist[i] = 1.0 # trivial
    for j in range(i+1, N_samples):
        valid = np.where(np.logical_and(good_preds[i,:], good_preds[j,:]))[0]
        if len(valid)>0:
            same = X_leaves[i,valid] == X_leaves[j,valid]
            dist[j] = 1 - (np.sum(same) / len(valid))
    return (i, dist)

def return_synthetic_data(X, seed=42):
    """
    The function returns a matrix with the same dimensions as X but with synthetic data
    based on the marginal distributions of its features.
    The spans of each line are the same as the original data.
    """
    rng = np.random.default_rng(seed)
    X_syn = np.nan*np.ones(X.shape)
    for i in progressbar(range(X.shape[1])): # for each feature
        obs_vec = X[:,i]
        valid = np.argwhere(~np.isnan(obs_vec))[:,0]
        N = len(valid)
        if N>0:
            X_syn[valid,i] = rng.choice(obs_vec[valid], N)

    return X_syn
    
def gaussian_pdf(mu,sigma,x):
    return (np.exp(-((x-mu)**2)/(2*(sigma**2))))/(np.sqrt(2*np.pi)*sigma)
    
def return_synthetic_data_shift(X, mu_shift, sigma_shift, seed=42):
    """
    This version of return_synthetic_data, creates synthetic data by shifting the original data
    by a random shift with a probability distribution of two gaussians, with mean of +/-mu_shift
    and standard deviation of sigma_shift.
    """
    rng = np.random.default_rng(seed)
    
    x = np.arange(-mu_shift-2*sigma_shift,1+mu_shift+2*sigma_shift)
    pdf = gaussian_pdf(-mu_shift,sigma_shift,x)+gaussian_pdf(mu_shift,sigma_shift,x)
    pdf = pdf/np.sum(pdf)
    
    X_syn = np.nan*np.ones(X.shape)
    N = X.shape[0]
    L = X.shape[1]
    shifts = rng.choice(x, size=N, p=pdf)
    for i in progressbar(range(N)): # for each sample
        X_syn[i,max(0,shifts[i]):min(L,L+shifts[i])] = X[i,max(0,-shifts[i]):min(L,L-shifts[i])]
    
    return X_syn

def fix_nan_shifts(X, L):
    """
    return_synthetic_data creates nans at the edges. this function replaces it with a median.
    """
    
    N_samples = X.shape[0]
    N_features = X.shape[1]

    # get the limits of every sample
    X_valid = ~np.isnan(X)
    sample_i_start = np.argmax(X_valid, axis=1)
    sample_i_end = X.shape[1] - np.argmax(np.fliplr(X_valid), axis=1) # non inclusive
    for i in range(N_samples):
        left = np.median(X[i,sample_i_start[i]:(sample_i_start[i]+L)])
        right = np.median(X[i,(sample_i_end[i]-L):sample_i_end[i]])
        X[i,:sample_i_start[i]] = left
        X[i,sample_i_end[i]:] = right
    return X

def scale_data(X, scale=0.0, seed=42):
    """
    scales the data by a random factor which is distributed normally with mean 1 and std=scale
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(1-3*scale,max(1+3*scale,0.1),1000)
    pdf = gaussian_pdf(1,scale,x)
    pdf = pdf/np.sum(pdf)
    N = X.shape[0]
    C = rng.choice(x, size=N, p=pdf)
    return C.reshape(-1,1)*X

def log(path, **kwargs):
    with open(path, 'w') as f:
        for key,value in kwargs.items():
            f.write('{0} = {1}\n'.format(key,value))























