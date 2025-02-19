import keras
import numpy as np
from ..layers import KDMLayer, RBFKernelLayer
from ..utils import pure2dm, dm2discrete
from sklearn.metrics import pairwise_distances


class KDMSequentialClassModel(keras.Model):
    def __init__(self,
                 encoded_size,
                 dim_y,
                 encoder,
                 n_comp,
                 sigma=0.1,
                 sequence=[],
                 **kwargs):
        super().__init__(**kwargs)
        self.dim_y = dim_y
        self.encoded_size = encoded_size
        self.encoder = encoder
        self.n_comp = n_comp
        input = KDMLayer(kernel=RBFKernelLayer(sigma=sigma,
                                               dim=encoded_size,
                                               trainable=True),
                         dim_x=encoded_size,
                         dim_y=dim_y,
                         n_comp=n_comp)
        model_sequence = [input]
        for layer in sequence:
            model_sequence.append(
                KDMLayer(kernel=layer['kernel'],
                         dim_x=layer['dim_x'],
                         dim_y=layer['dim_y'],
                         n_comp=layer['n_comp']
                         )
            )
        self.model = keras.Sequential(
            model_sequence
        )
        print(self.model)

    def call(self, input):
        encoded = self.encoder(input)
        rho_x = pure2dm(encoded)
        rho_y = self.model(rho_x)
        probs = dm2discrete(rho_y)
        return probs

    def init_components(self, samples_x, samples_y, init_sigma=False, sigma_mult=1, index=0):
        encoded_x = self.encoder(samples_x)
        print(index)
        if init_sigma:
            np_encoded_x = keras.ops.convert_to_numpy(encoded_x)
            distances = pairwise_distances(np_encoded_x)
            sigma = np.mean(distances) * sigma_mult
            if (index == 0):
                self.model.layers[index].kernel.sigma.assign(sigma)
        print(self.model.layers)
        self.model.layers[index].c_x.assign(encoded_x)
        self.model.layers[index].c_y.assign(samples_y)
        self.model.layers[index].c_w.assign(
            keras.ops.ones((self.model.layers[index].n_comp,)) / self.model.layers[index].n_comp)
