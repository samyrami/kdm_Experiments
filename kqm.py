import numpy as np
import tensorflow as tf
import numpy as np
import keras

import tensorflow_probability as tfp
tfd = tfp.distributions


def dm2comp(dm):
    '''
    Extract vectors and weights from a factorized density matrix representation
    Arguments:
     dm: tensor of shape (bs, n, d + 1)
    Returns:
     w: tensor of shape (bs, n)
     v: tensor of shape (bs, n, d)
    '''
    return dm[:, :, 0], dm[:, :, 1:]


def comp2dm(w, v):
    '''
    Construct a factorized density matrix from vectors and weights
    Arguments:
     w: tensor of shape (bs, n)
     v: tensor of shape (bs, n, d)
    Returns:
     dm: tensor of shape (bs, n, d + 1)
    '''
    return tf.concat((w[:, :, tf.newaxis], v), axis=2)

def samples2dm(samples):
    '''
    Construct a factorized density matrix from a batch of samples
    each sample will have the same weight. Samples that are all 
    zero will be ignored.
    Arguments:
        samples: tensor of shape (bs, n, d)
    Returns:
        dm: tensor of shape (bs, n, d + 1)
    '''
    w = tf.reduce_any(samples, axis=-1)
    w = w / tf.reduce_sum(w, axis=-1, keepdims=True)
    return comp2dm(w, samples)

def pure2dm(psi):
    '''
    Construct a factorized density matrix to represent a pure state
    Arguments:
     psi: tensor of shape (bs, d)
    Returns:
     dm: tensor of shape (bs, 1, d + 1)
    '''
    ones = tf.ones_like(psi[:, 0:1])
    dm = tf.concat((ones[:,tf.newaxis, :],
                    psi[:,tf.newaxis, :]),
                   axis=2)
    return dm


def dm2discrete(dm):
    '''
    Creates a discrete distribution from the components of a density matrix
    Arguments:
     dm: tensor of shape (bs, n, d + 1)
    Returns:
     prob: vector of probabilities (bs, d)
    '''
    w, v = dm2comp(dm)
    w = w / tf.reduce_sum(w, axis=-1, keepdims=True)
    norms_v = tf.expand_dims(tf.linalg.norm(v, axis=-1), axis=-1)
    v = v / norms_v
    probs = tf.einsum('...j,...ji->...i', w, v ** 2, optimize="optimal")
    return probs


def dm2distrib(dm, sigma):
    '''
    Creates a Gaussian mixture distribution from the components of a density
    matrix with an RBF kernel 
    Arguments:
     dm: tensor of shape (bs, n, d + 1)
     sigma: sigma parameter of the RBF kernel 
    Returns:
     gm: mixture of Gaussian distribution with shape (bs, )
    '''
    w, v = dm2comp(dm)
    gm = tfd.MixtureSameFamily(reparameterize=True,
            mixture_distribution=tfd.Categorical(
                                    probs=w),
            components_distribution=tfd.Independent( tfd.Normal(
                    loc=v,  # component 2
                    scale=sigma / np.sqrt(2.)),
                    reinterpreted_batch_ndims=1))
    return gm


def pure_dm_overlap(x, dm, kernel):
    '''
    Calculates the overlap of a state  \phi(x) with a density 
    matrix in a RKHS defined by a kernel
    Arguments:
      x: tensor of shape (bs, d)
     dm: tensor of shape (bs, n, d + 1)
     kernel: kernel function 
              k: (bs, d) x (bs, n, d) -> (bs, n)
    Returns:
     overlap: tensor with shape (bs, )
    '''
    w, v = dm2comp(dm)
    overlap = tf.einsum('...i,...i->...', w, kernel(x, v) ** 2)
    return overlap

## Kernels

class CompTransKernelLayer(tf.keras.layers.Layer):
    def __init__(self, transform, kernel):
        '''
        Composes a transformation and a kernel to create a new
        kernel.
        Arguments:
            transform: a function f that transform the input before feeding it to the 
                    kernel
                    f:(bs, d) -> (bs, D) 
            kernel: a kernel function
                    k:(bs, n, D)x(m, D) -> (bs, n, m)
        '''
        super(CompTransKernelLayer, self).__init__()
        self.transform = transform
        self.kernel = kernel

    def call(self, A, B):
        '''
        Input:
            A: tensor of shape (bs, n, d)
            B: tensor of shape (m, d)
        Result:
            K: tensor of shape (bs, n, m)
        '''
        shape = tf.shape(A) # (bs, n, d)
        A = tf.reshape(A, [shape[0] * shape[1], shape[2]])
        A = self.transform(A)
        dim_out = tf.shape(A)[1]
        A = tf.reshape(A, [shape[0], shape[1], dim_out])
        B = self.transform(B)
        return self.kernel(A, B)
    
    def log_weight(self):
        return self.kernel.log_weight()

class RBFKernelLayer(tf.keras.layers.Layer):
    def __init__(self, sigma, dim, trainable=True, name=None):
        '''
        Builds a layer that calculates the rbf kernel between two set of vectors
        Arguments:
            sigma: RBF scale parameter. If it is a tf.Variable it will be used as is.
                     Otherwise it will create a trainable variable with the given value.
        '''
        super(RBFKernelLayer, self).__init__(name=name)
        if type(sigma) is tf.Variable:
            self.sigma = sigma
        else:
            self.sigma = tf.Variable(sigma, dtype=tf.float32, trainable=trainable)
        self.dim = dim

    def call(self, A, B):
        '''
        Input:
            A: tensor of shape (bs, n, d)
            B: tensor of shape (m, d)
        Result:
            K: tensor of shape (bs, n, m)
        '''
        shape_A = tf.shape(A)
        shape_B = tf.shape(B)
        A_norm = tf.norm(A, axis=-1)[..., tf.newaxis] ** 2
        B_norm = tf.norm(B, axis=-1)[tf.newaxis, tf.newaxis, :] ** 2
        A_reshaped = tf.reshape(A, [-1, shape_A[2]])
        AB = tf.matmul(A_reshaped, B, transpose_b=True) 
        AB = tf.reshape(AB, [shape_A[0], shape_A[1], shape_B[0]])
        dist2 = A_norm + B_norm - 2. * AB
        dist2 = tf.clip_by_value(dist2, 0., np.inf)
        K = tf.exp(-dist2 / (2. * self.sigma ** 2.))
        return K
    
    def log_weight(self):
        return - self.dim * tf.math.log(tf.abs(self.sigma) + 1e-12) - self.dim * np.log(4 * np.pi) 



'''
Keras layer version of CosineKernel
''' 
class CosineKernelLayer(tf.keras.layers.Layer):
    def __init__(self):
        '''
        Builds a layer that calculates the cosine kernel between two set of vectors
        '''
        super(CosineKernelLayer, self).__init__()
        self.eps = 1e-6

    def call(self, A, B):
        '''
        Input:
            A: tensor of shape (bs, n, d)
            B: tensor of shape (m, d)
        Result:
            K: tensor of shape (bs, n, m)
        '''
        A = tf.math.divide_no_nan(A, 
                                  tf.expand_dims(tf.norm(A, axis=-1), axis=-1))
        B = tf.math.divide_no_nan(B, 
                                  tf.expand_dims(tf.norm(B, axis=-1), axis=-1))
        K = tf.einsum("...nd,md->...nm", A, B)
        return K
    
    def log_weight(self):
        return 0


class CrossProductKernelLayer(tf.keras.layers.Layer):

    def __init__(self, dim1, kernel1, kernel2):
        '''
        Create a layer that calculates the cross product kernel of two input
        kernels. The input vector are divided into two parts, the first of dimension 
        dim1 and the second of dimension d - dim1. Each input kernel is applied to 
        one of the parts of the input. 
        Arguments:
            dim1: the dimension of the first part of the input vector
            kernel1: a kernel function
                    k1:(bs, n, dim1)x(m, dim1) -> (bs, n, m)
            kernel2: a kernel function
                    k2:(bs, n, d - dim1)x(m, d - dim1) -> (bs, n, m)
        '''

        super(CrossProductKernelLayer, self).__init__()
        self.dim1 = dim1
        self.kernel1 = kernel1
        self.kernel2 = kernel2

    def call(self, A, B):
        '''
        Input:
            A: tensor of shape (bs, n, d)
            B: tensor of shape (m, d)
        Result:
            K: tensor of shape (bs, n, m)
        '''
        A1 = A[:, :, :self.dim1]
        A2 = A[:, :, self.dim1:]
        B1 = B[:, :self.dim1]
        B2 = B[:, self.dim1:]
        return self.kernel1(A1, B1) * self.kernel2(A2, B2)
    
    def log_weight(self):
        return self.kernel1.log_weight() + self.kernel2.log_weight()

## Layers and models

def l1_loss(vals):
    '''
    Calculate the l1 loss for a batch of vectors
    Arguments:
        vals: tensor with shape (b_size, n)
    '''
    b_size = tf.cast(tf.shape(vals)[0], dtype=tf.float32)
    vals = vals / tf.norm(vals, axis=1)[:, tf.newaxis]
    loss = tf.reduce_sum(tf.abs(vals)) / b_size
    return loss

class KQMUnit(tf.keras.layers.Layer):
    """Kernel Quantum Measurement Unit
    Receives as input a factored density matrix represented by a set of vectors
    and weight values. 
    Returns a resulting factored density matrix.
    Input shape:
        (batch_size, n_comp_in, dim_x + 1)
        where dim_x is the dimension of the input state
        and n_comp_in is the number of components of the input factorization. 
        The weights of the input factorization of sample i are [i, :, 0], 
        and the vectors are [i, :, 1:dim_x + 1].
    Output shape:
        (batch_size, n_comp, dim_y)
        where dim_y is the dimension of the output state
        and n_comp is the number of components used to represent the train
        density matrix. The weights of the
        output factorization for sample i are [i, :, 0], and the vectors
        are [i, :, 1:dim_y + 1].
    Arguments:
        dim_x: int. the dimension of the input state
        dim_y: int. the dimension of the output state
        x_train: bool. Whether to train or not the x compoments of the train
                       density matrix.
        x_train: bool. Whether to train or not the y compoments of the train
                       density matrix.
        w_train: bool. Whether to train or not the weights of the compoments 
                       of the train density matrix. 
        n_comp: int. Number of components used to represent 
                 the train density matrix
        l1_act: float. Coefficient of the regularization term penalizing the l1
                       norm of the activations.
        l1_x: float. Coefficient of the regularization term penalizing the l1
                       norm of the x components.
        l1_y: float. Coefficient of the regularization term penalizing the l1
                       norm of the y components.
    """
    def __init__(
            self,
            kernel,
            dim_x: int,
            dim_y: int,
            x_train: bool = True,
            y_train: bool = True,
            w_train: bool = True,
            n_comp: int = 0, 
            l1_x: float = 0.,
            l1_y: float = 0.,
            l1_act: float = 0.,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.kernel = kernel
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.x_train = x_train
        self.y_train = y_train
        self.w_train = w_train
        self.n_comp = n_comp
        self.l1_x = l1_x
        self.l1_y = l1_y
        self.l1_act = l1_act

    def build(self, input_shape):
        if (input_shape[1] and input_shape[2] != self.dim_x + 1 
            or len(input_shape) != 3):
            raise ValueError(
                f'Input dimension must be (batch_size, m, {self.dim_x + 1} )'
                f' but it is {input_shape}'
                )
        self.c_x = self.add_weight(
            "c_x",
            shape=(self.n_comp, self.dim_x),
            #initializer=tf.keras.initializers.orthogonal(),
            initializer=tf.keras.initializers.random_normal(),
            trainable=self.x_train)
        self.c_y = self.add_weight(
            "c_y",
            shape=(self.n_comp, self.dim_y),
            initializer=tf.keras.initializers.Constant(np.sqrt(1./self.dim_y)),
            #initializer=tf.keras.initializers.random_normal(),
            trainable=self.y_train)
        self.comp_w = self.add_weight(
            "comp_w",
            shape=(self.n_comp,),
            initializer=tf.keras.initializers.constant(1./self.n_comp),
            trainable=self.w_train) 
        self.eps = 1e-10
        self.built = True

    def call(self, inputs):
        
        # Weight regularizers
        if self.l1_x != 0:
            self.add_loss(self.l1_x * l1_loss(self.c_x))
        if self.l1_y != 0:
            self.add_loss(self.l1_y * l1_loss(self.c_y))
        comp_w = tf.clip_by_value(self.comp_w, 1e-10, 1)
        # normalize comp_w to sum to 1
        comp_w = comp_w / tf.reduce_sum(comp_w)
        in_w = inputs[:, :, 0]  # shape (b, n_comp_in)
        in_v = inputs[:, :, 1:] # shape (b, n_comp_in, dim_x)
        out_vw = self.kernel(in_v, self.c_x)  # shape (b, n_comp_in, n_comp)
        out_w = (tf.expand_dims(tf.expand_dims(comp_w, axis=0), axis=0) *
                 tf.square(out_vw)) # shape (b, n_comp_in, n_comp)
        out_w = tf.maximum(out_w, self.eps) #########
        # out_w_sum = tf.maximum(tf.reduce_sum(out_w, axis=2), self.eps)  # shape (b, n_comp_in)
        out_w_sum = tf.reduce_sum(out_w, axis=2) # shape (b, n_comp_in)
        out_w = out_w / tf.expand_dims(out_w_sum, axis=2)
        out_w = tf.einsum('...i,...ij->...j', in_w, out_w, optimize="optimal")
                # shape (b, n_comp)
        if self.l1_act != 0:
            self.add_loss(self.l1_act * l1_loss(out_w))
        out_w = tf.expand_dims(out_w, axis=-1) # shape (b, n_comp, 1)
        out_y_shape = tf.shape(out_w) + tf.constant([0, 0, self.dim_y - 1])
        out_y = tf.broadcast_to(tf.expand_dims(self.c_y, axis=0), out_y_shape)
        out = tf.concat((out_w, out_y), 2)
        return out

    def get_config(self):
        config = {
            "dim_x": self.dim_x,
            "dim_y": self.dim_y,
            "n_comp": self.n_comp,
            "x_train": self.x_train,
            "y_train": self.y_train,
            "w_train": self.w_train,
            "l1_x": self.l1_x,
            "l1_y": self.l1_y,
            "l1_act": self.l1_act,
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def compute_output_shape(self, input_shape):
        return (self.dim_y + 1, self.n_comp)

class KQMOverlap(tf.keras.layers.Layer):
    """Kernel Quantum Measurement Overlap Unit
    Receives as input a vector and calculates its overlap with the unit density 
    matrix.
    Input shape:
        (batch_size, dim_x)
        where dim_x is the dimension of the input state
    Output shape:
        (batch_size, )
    Arguments:
        kernel: a kernel function
        dim_x: int. the dimension of the input state
        x_train: bool. Whether to train the or not the compoments of the train
                       density matrix.
        w_train: bool. Whether to train the or not the weights of the compoments 
                       of the train density matrix. 
        n_comp: int. Number of components used to represent 
                 the train density matrix
    """

    def __init__(
            self,
            kernel,
            dim_x: int,
            x_train: bool = True,
            w_train: bool = True,
            n_comp: int = 0, 
            **kwargs
    ):
        super().__init__(**kwargs)
        self.kernel = kernel
        self.dim_x = dim_x
        self.x_train = x_train
        self.w_train = w_train
        self.n_comp = n_comp

    def build(self, input_shape):
        if (input_shape[1] != self.dim_x
            or len(input_shape) != 2):
            raise ValueError(
                f'Input dimension must be (batch_size, {self.dim_x} )'
                f' but it is {input_shape}'
                )
        self.c_x = self.add_weight(
            "c_x",
            shape=(self.n_comp, self.dim_x),
            #initializer=tf.keras.initializers.orthogonal(),
            initializer=tf.keras.initializers.random_normal(),
            trainable=self.x_train)
        self.comp_w = self.add_weight(
            "comp_w",
            shape=(self.n_comp,),
            initializer=tf.keras.initializers.constant(1./self.n_comp),
            trainable=self.w_train) 
        self.built = True

    def call(self, inputs):
        comp_w = tf.clip_by_value(self.comp_w, 1e-10, 1)
        # normalize comp_w to sum to 1
        comp_w = comp_w / tf.reduce_sum(comp_w)
        in_v = inputs[:, tf.newaxis, :]
        out_vw = self.kernel(in_v, self.c_x) ** 2 # shape (b, 1, n_comp)
        out_w = tf.einsum('...j,...ij->...', comp_w, out_vw, optimize="optimal")
        return out_w

    def get_config(self):
        config = {
            "dim_x": self.dim_x,
            "n_comp": self.n_comp,
            "x_train": self.x_train,
            "w_train": self.w_train,
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def compute_output_shape(self, input_shape):
        return (1,)

class KQMClassModel(tf.keras.Model):
    def __init__(self,
                 dim_x,
                 dim_y,
                 sigma,
                 n_comp,
                 x_train=True):
        super().__init__()
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.n_comp = n_comp
        self.kernel_x = RBFKernelLayer(sigma, dim=dim_x)
        self.kqmu = KQMUnit(self.kernel_x,
                            dim_x=dim_x,
                            dim_y=dim_y,
                            n_comp=n_comp,
                            x_train=x_train)

    def call(self, inputs):
        rho_x = pure2dm(inputs)
        rho_y = self.kqmu(rho_x)
        probs = dm2discrete(rho_y)
        return probs

class BagKQMClassModel(tf.keras.Model):
    def __init__(self,
                 dim_x,
                 dim_y,
                 sigma,
                 n_comp,
                 x_train=True,
                 l1_y=0.):
        super().__init__()
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.n_comp = n_comp
        kernel_x = RBFKernelLayer(sigma)
        self.kqmu = KQMUnit(kernel_x,
                            dim_x=dim_x,
                            dim_y=dim_y,
                            n_comp=n_comp,
                            x_train=x_train,
                            l1_y=l1_y)

    def call(self, inputs):
        in_shape = tf.shape(inputs)
        w = tf.ones_like(inputs[:, :, 0]) / in_shape[1]
        rho_x = comp2dm(w, inputs)
        rho_y = self.kqmu(rho_x)
        probs = dm2discrete(rho_y)
        return rho_y

class KQMDenEstModel(tf.keras.Model):
    def __init__(self,
                 dim_x,
                 sigma,
                 n_comp):
        super().__init__()
        self.dim_x = dim_x
        self.n_comp = n_comp
        self.kernel = RBFKernelLayer(sigma, dim=dim_x)
        self.kqmover = KQMOverlap(self.kernel,
                                dim_x=dim_x,
                                n_comp=n_comp)

    def call(self, inputs):
        log_probs = (tf.math.log(self.kqmover(inputs) + 1e-12)
                     + self.kernel.log_weight())
        self.add_loss(-tf.reduce_mean(log_probs))
        return log_probs


class KQMDenEstModel2(tf.keras.Model):
    def __init__(self,
                 dim_x,
                 dim_y,
                 sigma,
                 n_comp,
                 trainable_sigma=True):
        super().__init__()
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.n_comp = n_comp
        self.kernel_x = RBFKernelLayer(sigma, dim=dim_x, trainable=trainable_sigma)
        self.kernel_y = CosineKernelLayer()
        self.kernel = CrossProductKernelLayer(dim1=dim_x, kernel1=self.kernel_x, kernel2=self.kernel_y)
        self.kqmover = KQMOverlap(self.kernel,
                                dim_x=dim_x + dim_y,
                                n_comp=n_comp)

    def call(self, inputs):
        log_probs = (tf.math.log(self.kqmover(inputs) + 1e-12)
                     + self.kernel.log_weight())
        self.add_loss(-tf.reduce_mean(log_probs))
        return log_probs


class KQMDenEstModel3(tf.keras.Model):
    def __init__(self,
                 dim_x,
                 dim_y,
                 n_comp):
        super().__init__()
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.n_comp = n_comp
        self.kernel_x = CosineKernelLayer()
        self.kernel_y = CosineKernelLayer()
        self.kernel = CrossProductKernelLayer(dim1=dim_x, kernel1=self.kernel_x, kernel2=self.kernel_y)
        self.kqmover = KQMOverlap(self.kernel,
                                dim_x=dim_x + dim_y,
                                n_comp=n_comp)

    def call(self, inputs):
        log_probs = (tf.math.log(self.kqmover(inputs) + 1e-12)
                     + self.kernel.log_weight())
        self.add_loss(-tf.reduce_mean(log_probs))
        return log_probs
