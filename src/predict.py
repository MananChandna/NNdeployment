import pandas as pd
import numpy as np

from src.config import config
import src.preprocessing.preprocessors as pp
from src.preprocessing.data_management import load_dataset, save_model, load_model
from src.train_pipeline import layer_neurons_weighted_sum, layer_neurons_output

h = [None]*config.NUM_LAYERS
z = [None]*config.NUM_LAYERS


def inference(X, trained_biases, trained_weights):  
    num_samples = X.shape[0]
    h = [None] * config.NUM_LAYERS
    z = [None] * config.NUM_LAYERS

    predictions = []

    for i in range(num_samples):
        h[0] = X[i].reshape(1, -1)

        for l in range(1, config.NUM_LAYERS):
            z[l] = layer_neurons_weighted_sum(h[l-1], trained_biases[l], trained_weights[l])
            #print("z[{}].shape = {}".format(l, z[l].shape))
            h[l] = layer_neurons_output(z[l], config.f[l])
            #print("h[{}].shape = {}".format(l, h[l].shape))

        predictions.append(int(h[config.NUM_LAYERS - 1] > 0.5))

    return np.array(predictions)