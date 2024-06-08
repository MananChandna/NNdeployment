import pandas as pd
import numpy as np

from src.config import config
import src.preprocessing.preprocessors as pp
from src.preprocessing.data_management import load_dataset, save_model, load_model

import train_pipeline as tp 


def layer_neurons_weighted_sum(previous_layer_neurons_outputs, current_layer_neurons_biases, current_layer_neurons_weights):
    return current_layer_neurons_biases + np.matmul(previous_layer_neurons_outputs, current_layer_neurons_weights)

def layer_neurons_output(current_layer_neurons_weighted_sums, current_layer_neurons_activation_function):
    if current_layer_neurons_activation_function == "linear":
        return current_layer_neurons_weighted_sums
    elif current_layer_neurons_activation_function == "sigmoid":
        return 1/(1 + np.exp(-current_layer_neurons_weighted_sums))
    elif current_layer_neurons_activation_function == "tanh":
        return (np.exp(current_layer_neurons_weighted_sums) - np.exp(-current_layer_neurons_weighted_sums)) / (np.exp(current_layer_neurons_weighted_sums) + np.exp(-current_layer_neurons_weighted_sums))
    elif current_layer_neurons_activation_function == "relu":
        return current_layer_neurons_weighted_sums * (current_layer_neurons_weighted_sums > 0)

def predict(input_data, model):
    """
    Make a prediction using the trained model.
    """
    # Preprocess the input data
    preprocessor = preprocess_data()
    preprocessor.fit(np.array(input_data).reshape(1, -1))  # Ensure the input data is 2D
    preprocessed_data, _ = preprocessor.transform(X=np.array(input_data).reshape(1, -1))
    
    # Extract model parameters
    theta0 = model['params']['biases']
    theta = model['params']['weights']
    f = model['activations']
    NUM_LAYERS = len(theta0)
    
    # Forward propagation
    h = [None] * NUM_LAYERS
    z = [None] * NUM_LAYERS
    
    h[0] = preprocessed_data
    for l in range(1, NUM_LAYERS):
        z[l] = layer_neurons_weighted_sum(h[l-1], theta0[l], theta[l])
        h[l] = layer_neurons_output(z[l], f[l])
    
    # The final output
    return h[NUM_LAYERS-1]

if __name__ == "__main__":
    # Example input data
    input_data = [0, 1]  # Replace with actual input data
    
    # Load the model
    model_filepath = "two_input_xor_nn.pkl"  # Ensure this matches the saved model filename
    model = load_model(model_filepath)
    
    # Make a prediction
    prediction = predict(input_data, model)
    print("Prediction:", prediction)