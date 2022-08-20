# imports
import numpy as np
import tensorflow as tf
from tensorflow.keras import applications
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from tensorflow.keras import Model
from tensorflow.keras import utils

# ==================================================================================

class DistillationDataGenerator(utils.Sequence):
    
    def __init__(self, X, D, batch_size=32, shuffle=False, seed=42, snr_range=None, full_epoch=False, norm=True):
        
        # saving arguments
        self.X = X.astype(np.float32)
        self.D = D.astype(np.float32)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.full_epoch = full_epoch
        self.norm = norm
        if snr_range!=None:
            self.noise = True
            self.snr_range = snr_range
        else:
            self.noise = False
        
        # saving data dimensions
        self.N_samples = X.shape[0]
        self.N_features = X.shape[1]
        
        # shuffling
        self.rng = np.random.default_rng(seed)
        if self.full_epoch:
            #self.couples = np.transpose([np.tile(np.arange(self.N_samples), self.N_samples), np.repeat(np.arange(self.N_samples), self.N_samples)])
            self.couples = np.stack(np.triu_indices(self.N_samples), axis=1)
        self.on_epoch_end()

    def __len__(self):
        if self.full_epoch:
            return int(np.ceil(len(self.couples) / self.batch_size))
        else:
            return int(np.ceil(self.N_samples / self.batch_size))

    def __getitem__(self, index):
        
        if self.full_epoch:
            couples = self.couples[index*self.batch_size:(index+1)*self.batch_size,:]
            x_indices = couples[:,0]
            y_indices = couples[:,1]
        else:
            x_indices = self.rng.integers(0, self.N_samples, self.batch_size)
            y_indices = self.rng.integers(0, self.N_samples, self.batch_size)
        
        # get the samples
        x = self.X[x_indices,:]
        y = self.X[y_indices,:]
        d = self.D[x_indices,y_indices]
        
        # add noise
        if self.noise:
            snr = np.random.uniform(low=self.snr_range[0], high=self.snr_range[1], size=None)
            N_std_x = np.sqrt((np.std(x,axis=1)**2)/snr)
            x += N_std_x.reshape(-1,1)*np.random.randn(x.shape[0],x.shape[1])
            N_std_y = np.sqrt((np.std(y,axis=1)**2)/snr)
            y += N_std_y.reshape(-1,1)*np.random.randn(y.shape[0],y.shape[1])
        
        # normalize
        if self.norm:
            xy_min = np.stack((x.min(axis=1),y.min(axis=1)),axis=1).min(axis=1).reshape(-1,1)
            x -= xy_min
            y -= xy_min
            xy_max = np.stack((x.max(axis=1),y.max(axis=1)),axis=1).max(axis=1).reshape(-1,1)
            x /= xy_max
            y /= xy_max
        
        return (x,y),d

    def on_epoch_end(self):
        if self.full_epoch and self.shuffle:
            # shuffle the indices on epoch end
            self.rng.shuffle(self.couples, axis=0)

# ==================================================================================

class DistanceLayer(layers.Layer):
    """
    This layer is responsible for computing the distance
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x, y):
        dist = tf.reduce_sum(tf.square(x - y), -1)
        return dist
        
class NormalizeDistance(layers.Layer):
    """
    This layer normalizes the distance to [0,1]
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x):
        dist = (x+1)/2
        return dist

# ==================================================================================

def L1(d, d_hat):
    return tf.reduce_mean(tf.abs(d - d_hat), -1)
    
def L2(d, d_hat):
    return tf.reduce_mean(tf.square(d - d_hat), -1)

class SiameseModel(Model):
    """
    The Siamese Network model with a custom training and testing loops.
    """

    def __init__(self, siamese_network, dist_loss='L1'):
        super(SiameseModel, self).__init__()
        self.siamese_network = siamese_network
        self.loss_tracker = metrics.Mean(name="loss")
        self.dist_loss = dist_loss
        if dist_loss=='L1':
            self.loss_func = L1
        if dist_loss=='L2':
            self.loss_func = L2

    def call(self, inputs):
        #print('<<<call>>>: inputs is {0} of len {1}'.format(str(type(inputs)), str(len(inputs))))
        return self.siamese_network(inputs)

    def train_step(self, data):
        # GradientTape is a context manager that records every operation that
        # you do inside. We are using it here to compute the loss so we can get
        # the gradients and apply them using the optimizer specified in
        # `compile()`.
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)

        # Storing the gradients of the loss function with respect to the
        # weights/parameters.
        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)

        # Applying the gradients on the model using the specified optimizer
        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights)
        )

        # Let's update and return the training loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self._compute_loss(data)

        # Let's update and return the loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def _compute_loss(self, data):
    
        # data is a tuple of the 2 spectra x and y, and the RF's distance d
        #x, y, d = data
        #print('<<<_compute_loss>>>: data is {0} of len {1}'.format(str(type(data)), str(len(data))))
        inputs, d = data # bugfix
        #print('<<<_compute_loss>>>: inputs is {0} of len {1}'.format(str(type(inputs)), str(len(inputs))))
        #print('<<<_compute_loss>>>: d is {0} of len {1}'.format(str(type(d)), str(d.shape)))
    
        # The output of the network is the distance
        #d_hat = self.siamese_network(data)
        d_hat = self.siamese_network(inputs) # bugfix

        # the loss is either L1 or L2 loss between the vectors
        loss = self.loss_func(d, d_hat)
        
        return loss

    @property
    def metrics(self):
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [self.loss_tracker]
