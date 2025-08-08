import pandas as pd
import numpy as np
import math

import jax
import jax.numpy as jnp
from jax import random, grad
from jax.nn import relu, sigmoid
from functools import partial

import matplotlib.pyplot as plt

import pytz
import os
import sys
import pickle
from datetime import datetime

# Define network parameters
def initialize_random_weights(mean, std, shape = ()):
    return np.random.normal(loc=mean, scale=std, size=shape)
    
# We use a 2-degree polynomial function to approximate 0 and 1/2 to 0 and 1 stays in 1
def polynomial_function(x):
    return 2 * (x ** 2) - x

# We use a sinusoidal function to approximate odd numbers by their immediately preceding even number and preserve differentiability
def lower_even(x):
    return x - 0.5 * (1 - jnp.cos(jnp.pi * x))

# We use a sinusoidal function to approximate 0 for evens and 1 for odds while preserving differentiability
def differentiable_even_or_odd(x):
    return 0.5 * (1 - jnp.cos(jnp.pi * x))
    
# Function to generate dataset with multiplication
def generate_dataset_with_zeros(size, n_max=10):
    # Generate two columns of random numbers between 0 and 9
    column_1 = np.random.randint(0, n_max, size)
    column_2 = np.random.randint(0, n_max, size)

    # Create a DataFrame with the two columns
    dataset = pd.DataFrame({
        'Column_1': column_1,
        'Column_2': column_2
    })

    # Create the third column by multiplying the first two
    dataset['Column_3'] = dataset['Column_1'] * dataset['Column_2']

    return dataset

def generate_dataset_without_zeros(size, n_max=10):
    # Generate two columns of random numbers between 1 and 9
    column_1 = np.random.randint(1, n_max, size)
    column_2 = np.random.randint(1, n_max, size)

    # Create a DataFrame with the two columns
    dataset = pd.DataFrame({
        'Column_1': column_1,
        'Column_2': column_2
    })

    # Create the third column by multiplying the first two
    dataset['Column_3'] = dataset['Column_1'] * dataset['Column_2']

    return dataset

def generate_test_dataset(n_max=10):
    # Create the columns
    column_1 = list(range(n_max)) * n_max  # Numbers from 0 to 9 repeated 10 times
    column_2 = [i for i in range(n_max) for _ in range(n_max)]  # Numbers from 0 to 9 repeated sequentially 10 times

    # Create a DataFrame with the two columns
    dataset = pd.DataFrame({
        'Column_1': column_1,
        'Column_2': column_2,
    })

    # Create the third column by multiplying the first two
    dataset['Column_3'] = dataset['Column_1'] * dataset['Column_2']

    return dataset


def decimal_to_binary(n, bits):
    if 0 <= n < 2**bits:
        # Convert the number to a binary string and then to an array of integers (0 and 1)
        return np.array(list(format(n, f'0{bits}b'))).astype(np.int8)
    else:
        raise ValueError("Number out of range")

# Function to convert binary number to decimal
def binary_to_decimal(binary_vector, bits):
    # Ensure the vector has the correct number of elements
    if len(binary_vector) != bits:
        raise ValueError(f"The vector must have exactly {bits} elements.")

    # Calculate the decimal number
    decimal = 0
    for i in range(bits):
        decimal += binary_vector[i] * (2 ** (bits - 1 - i))

    return decimal

def transform_to_tridimensional_matrix(dataset, bits_init=4, bits_end=7):
    rows, cols = dataset.shape
    if cols != 3:
        raise ValueError("The dataset must have exactly 3 columns.")

    # Initialize the three matrices
    matrix_column_1 = np.zeros((rows, bits_init), dtype=np.int8)
    matrix_column_2 = np.zeros((rows, bits_init), dtype=np.int8)
    matrix_column_3 = np.zeros((rows, bits_end), dtype=np.int8)

    # Fill the matrices with the binary representation of each column
    for i in range(rows):
        matrix_column_1[i] = decimal_to_binary(dataset.iloc[i, 0], bits_init)
        matrix_column_2[i] = decimal_to_binary(dataset.iloc[i, 1], bits_init)
        matrix_column_3[i] = decimal_to_binary(dataset.iloc[i, 2], bits_end)

    return matrix_column_1, matrix_column_2, matrix_column_3
    
    
def prepare_dataset(level, size=1, couples_included=[]):       
    if level == -3:
        column_1 = []
        column_2 = []
        pairs = couples_included
        while len(column_1) < size:
            choice = pairs[np.random.choice(len(pairs))]
            column_1.append(choice[0])
            column_2.append(choice[1])
        dataset = pd.DataFrame({'Column_1': column_1,'Column_2': column_2,})
        dataset['Column_3'] = dataset['Column_1'] * dataset['Column_2']
        return dataset

    elif level == -2:
        dataset = generate_dataset_with_zeros(size)
        return dataset
        
    elif level == -1:
        dataset = generate_dataset_without_zeros(size)
        return dataset

    elif level == 0:
        couples_not_included = [(3, 3), (5, 5), (6, 6), (7, 7), (9, 9), (3, 6), (3, 7), (6, 3), (7, 3), (5, 7), (7, 5), (6, 7), (7, 6)]
        dataset = pd.DataFrame()
        while len(dataset) < size:
            column_1 = np.random.randint(1, 10, size)
            column_2 = np.random.randint(1, 10, size)
            temp_dataset = pd.DataFrame({'Column_1': column_1, 'Column_2': column_2})
            temp_dataset = temp_dataset[~temp_dataset[['Column_1', 'Column_2']].apply(tuple, axis=1).isin(couples_not_included)]
            dataset = pd.concat([dataset, temp_dataset])
        dataset = dataset.iloc[:size].reset_index(drop=True)
        dataset['Column_3'] = dataset['Column_1'] * dataset['Column_2']
        return dataset

    elif level == 1:
        column_1 = []
        column_2 = []
        pairs = [(5, 5), (9, 9)]
        while len(column_1) < size:
            choice = pairs[np.random.choice(len(pairs))]
            column_1.append(choice[0])
            column_2.append(choice[1])
        dataset = pd.DataFrame({'Column_1': column_1,'Column_2': column_2,})
        dataset['Column_3'] = dataset['Column_1'] * dataset['Column_2']
        return dataset

    elif level == 2:
        column_1 = []
        column_2 = []
        pairs = [(3, 3), (6, 6), (3, 6), (6, 3), (5, 7), (7, 5)]
        while len(column_1) < size:
            choice = pairs[np.random.choice(len(pairs))]
            column_1.append(choice[0])
            column_2.append(choice[1])
        dataset = pd.DataFrame({'Column_1': column_1,'Column_2': column_2,})
        dataset['Column_3'] = dataset['Column_1'] * dataset['Column_2']
        return dataset

    elif level == 3:
        column_1 = []
        column_2 = []
        pairs = [(3, 7), (6, 7), (7, 3), (7, 6)]
        while len(column_1) < size:
            choice = pairs[np.random.choice(len(pairs))]
            column_1.append(choice[0])
            column_2.append(choice[1])
        dataset = pd.DataFrame({'Column_1': column_1,'Column_2': column_2,})
        dataset['Column_3'] = dataset['Column_1'] * dataset['Column_2']
        return dataset

    elif level == 4:
        column_1 = [7] * size
        column_2 = [7] * size
        dataset = pd.DataFrame({'Column_1': column_1,'Column_2': column_2,})
        dataset['Column_3'] = dataset['Column_1'] * dataset['Column_2']
        return dataset

    elif level == 5:
        column_1 = []
        column_2 = []
        pairs = [(3, 3), (5, 5), (6, 6), (7, 7), (9, 9), (3, 6), (3, 7), (6, 3), (7, 3), (5, 7), (7, 5), (6, 7), (7, 6)]
        while len(column_1) < size:
            choice = pairs[np.random.choice(len(pairs))]
            column_1.append(choice[0])
            column_2.append(choice[1])
        dataset = pd.DataFrame({'Column_1': column_1,'Column_2': column_2,})
        dataset['Column_3'] = dataset['Column_1'] * dataset['Column_2']
        return dataset

    else:
        print('Bad index for the training stage.')
        return None

def prepare_outputs(stage, inputs_1, inputs_2, outputs_prev):
    if stage == 1:
        return np.array([np.outer(vec2, vec1).flatten() for vec1, vec2 in zip(inputs_1, inputs_2)])

    elif stage == 2:
        outputs = []
        matrix_step_2 = np.zeros((16, 28))
        for i in range(4):
            for j in range(4):
                matrix_step_2[i*4 + j, i*8 + j] = 1
        for vec1, vec2 in zip(inputs_1, inputs_2):
            outer_product = np.outer(vec2, vec1)
            flatten_vector = jnp.dot(outer_product.flatten(), matrix_step_2)
            outputs.append(flatten_vector)
        return np.array(outputs)

    elif stage == 3:
        outputs = []
        for vec1, vec2 in zip(inputs_1, inputs_2):
            outer_product = np.outer(vec2, vec1)
            z3 = lower_even(outer_product[2,3] + outer_product[3,2])
            z4 = lower_even(outer_product[1,3] + outer_product[2,2] + outer_product[3,1] + z3 * 1/2)
            z5 = lower_even(outer_product[0,3] + outer_product[1,2] + outer_product[2,1] + outer_product[3,0] + z4 * 1/2)
            z6 = lower_even(outer_product[0,2] + outer_product[1,1] + outer_product[2,0] + z5 * 1/2)
            z7 = lower_even(outer_product[0,1] + outer_product[1,0] + z6 * 1/2)
            outputs.append([z7, z6, z5, z4, z3, 0, 0])
        return np.array(outputs)

    elif stage == 4:
        return outputs_prev

    elif stage == 5:
        return outputs_prev

    else:
        print('Bad index for the training stage.')
        return None
        
# Perfect parameters needed for the stages where a part of the NN performs perfectly

# Create the W1_perfect matrix of zeros with dimensions (8,16)
W1_perfect = np.zeros((8, 16))
# Introduce the correct numbers to multiply
for k in range(4):
    for i in range(4):
        W1_perfect[i, 12+i-k*4] = 1/2
        W1_perfect[4+i, 4*i:4*i+4] = 1/2

# Create the W2_perfect matrix of zeros with dimensions (16,28)
W2_perfect = np.zeros((16, 28))

# Correctly place the 7-bit vectors
for i in range(4):
    for j in range(4):
        W2_perfect[i*4 + j, i*8 + j] = 1

# R vectors of dimension (28,1)
R3_perfect = np.zeros((28))
R4_perfect = np.zeros((28))
R5_perfect = np.zeros((28))
R6_perfect = np.zeros((28))
R7_perfect = np.zeros((28))

for i in range(4):
    R3_perfect[7*i + 5] = 1
    R4_perfect[7*i + 4] = 1
    R5_perfect[7*i + 3] = 1
    R6_perfect[7*i + 2] = 1
    R7_perfect[7*i + 1] = 1

# Scalar parameters v
v3_perfect = 1/2
v4_perfect = 1/2
v5_perfect = 1/2
v6_perfect = 1/2

# Matrix T of dimension (28,7)
T_perfect = np.zeros((28,7))

for i in range(7):
    for j in range(4):
        T_perfect[7*j + i, i] = 1

# Parameter v7
v7_perfect = 1/2

# Neural network in every stage

def neural_network_1(params, x1, x2):
    W1, h = params
    x = jnp.concatenate((x1, x2), axis=0)
    prev_vec = polynomial_function(jnp.dot(x, W1)) # Multiplies values, prev_vec is a (1,8) matrix
    #vec = jnp.dot(prev_vec, W2) # vec is a (1,28) dimensional vector
    #z3 = lower_even(relu(jnp.dot(vec, R3))) # z3 is a scalar with the second carry over
    #z4 = lower_even(relu(jnp.dot(vec, R4) + jnp.dot(z3, v3))) # z4 is a scalar with the third carry over
    #z5 = lower_even(relu(jnp.dot(vec, R5) + jnp.dot(z4, v4))) # z5 is a scalar with the fourth carry over
    #z6 = lower_even(relu(jnp.dot(vec, R6) + jnp.dot(z5, v5))) # z6 is a scalar with the fifth carry over
    #z7 = lower_even(relu(jnp.dot(vec, R7) + jnp.dot(z6, v6))) # z7 is a scalar with the seventh carry over
    #z = jnp.array([z7, z6, z5, z4, z3, 0, 0])
    #y = differentiable_even_or_odd(relu(jnp.dot(vec, T) + jnp.dot(z, v7)))
    return prev_vec

def neural_network_2(params, x1, x2):
    W2, h = params
    x = jnp.concatenate((x1, x2), axis=0)
    prev_vec = polynomial_function(jnp.dot(x, W1_perfect)) # Multiplies values, prev_vec is a (1,8) matrix
    vec = jnp.dot(prev_vec, W2) # vec is a (1,28) dimensional vector
    #z3 = lower_even(relu(jnp.dot(vec, R3))) # z3 is a scalar with the second carry over
    #z4 = lower_even(relu(jnp.dot(vec, R4) + jnp.dot(z3, v3))) # z4 is a scalar with the third carry over
    #z5 = lower_even(relu(jnp.dot(vec, R5) + jnp.dot(z4, v4))) # z5 is a scalar with the fourth carry over
    #z6 = lower_even(relu(jnp.dot(vec, R6) + jnp.dot(z5, v5))) # z6 is a scalar with the fifth carry over
    #z7 = lower_even(relu(jnp.dot(vec, R7) + jnp.dot(z6, v6))) # z7 is a scalar with the seventh carry over
    #z = jnp.array([z7, z6, z5, z4, z3, 0, 0])
    #y = differentiable_even_or_odd(relu(jnp.dot(vec, T) + jnp.dot(z, v7)))
    return vec

def neural_network_3(params, x1, x2):
    R3, R4, R5, R6, R7, v3, v4, v5, v6 = params
    x = jnp.concatenate((x1, x2), axis=0)
    prev_vec = polynomial_function(jnp.dot(x, W1_perfect)) # Multiplies values, prev_vec is a (1,8) matrix
    vec = jnp.dot(prev_vec, W2_perfect) # vec is a (1,28) dimensional vector
    z3 = lower_even(jnp.dot(vec, R3)) # z3 is a scalar with the second carry over
    z4 = lower_even(jnp.dot(vec, R4) + jnp.dot(z3, v3)) # z4 is a scalar with the third carry over
    z5 = lower_even(jnp.dot(vec, R5) + jnp.dot(z4, v4)) # z5 is a scalar with the fourth carry over
    z6 = lower_even(jnp.dot(vec, R6) + jnp.dot(z5, v5)) # z6 is a scalar with the fifth carry over
    z7 = lower_even(jnp.dot(vec, R7) + jnp.dot(z6, v6)) # z7 is a scalar with the seventh carry over
    #y = differentiable_even_or_odd(relu(jnp.dot(vec, T) + jnp.dot(z, v7)))
    return jnp.array([z7, z6, z5, z4, z3, 0, 0])

def neural_network_4(params, x1, x2):
    T, v7 = params
    x = jnp.concatenate((x1, x2), axis=0)
    prev_vec = polynomial_function(jnp.dot(x, W1_perfect)) # Multiplies values, prev_vec is a (1,8) matrix
    vec = jnp.dot(prev_vec, W2_perfect) # vec is a (1,28) dimensional vector
    z3 = lower_even(relu(jnp.dot(vec, R3_perfect))) # z3 is a scalar with the second carry over
    z4 = lower_even(relu(jnp.dot(vec, R4_perfect) + jnp.dot(z3, v3_perfect))) # z4 is a scalar with the third carry over
    z5 = lower_even(relu(jnp.dot(vec, R5_perfect) + jnp.dot(z4, v4_perfect))) # z5 is a scalar with the fourth carry over
    z6 = lower_even(relu(jnp.dot(vec, R6_perfect) + jnp.dot(z5, v5_perfect))) # z6 is a scalar with the fifth carry over
    z7 = lower_even(relu(jnp.dot(vec, R7_perfect) + jnp.dot(z6, v6_perfect))) # z7 is a scalar with the seventh carry over
    z = jnp.array([z7, z6, z5, z4, z3, 0, 0])
    y = differentiable_even_or_odd(relu(jnp.dot(vec, T) + jnp.dot(z, v7)))
    return y

def neural_network_5(params, x1, x2):
    W1, W2, R3, R4, R5, R6, R7, v3, v4, v5, v6, T, v7 = params
    x = jnp.concatenate((x1, x2), axis=0)
    prev_vec = polynomial_function(jnp.dot(x, W1)) # Multiplies values, prev_vec is a (1,8) matrix
    vec = jnp.dot(prev_vec, W2) # vec is a (1,28) dimensional vector
    z3 = lower_even(jnp.dot(vec, R3)) # z3 is a scalar with the second carry over
    z4 = lower_even(jnp.dot(vec, R4) + jnp.dot(z3, v3)) # z4 is a scalar with the third carry over
    z5 = lower_even(jnp.dot(vec, R5) + jnp.dot(z4, v4)) # z5 is a scalar with the fourth carry over
    z6 = lower_even(jnp.dot(vec, R6) + jnp.dot(z5, v5)) # z6 is a scalar with the fifth carry over
    z7 = lower_even(jnp.dot(vec, R7) + jnp.dot(z6, v6)) # z7 is a scalar with the seventh carry over
    z = jnp.array([z7, z6, z5, z4, z3, 0, 0])
    y = differentiable_even_or_odd(jnp.dot(vec, T) + jnp.dot(z, v7))
    return y

# Loss functions in every stage
def loss_1(params, x1, x2, y):
    pred = neural_network_1(params, x1, x2)
    return jnp.mean((pred - y)**2)

def loss_2(params, x1, x2, y):
    pred = neural_network_2(params, x1, x2)
    return jnp.mean((pred - y)**2)

def loss_3(params, x1, x2, y):
    pred = neural_network_3(params, x1, x2)
    return jnp.mean((pred - y)**2)

def loss_4(params, x1, x2, y):
    pred = neural_network_4(params, x1, x2)
    return jnp.mean((pred - y)**2)

def loss_5(params, x1, x2, y):
    pred = neural_network_5(params, x1, x2)
    return jnp.mean((pred - y)**2)


# Loss functions in every step
@jax.jit
def update_params_1(params, x1, x2, y, lr):
    gradients = grad(loss_1)(params, x1, x2, y)
    step_loss = loss_1(params, x1, x2, y)
    return [(p - lr * g) for p, g in zip(params, gradients)], step_loss

@jax.jit
def update_params_2(params, x1, x2, y, lr):
    gradients = grad(loss_2)(params, x1, x2, y)
    step_loss = loss_2(params, x1, x2, y)
    return [(p - lr * g) for p, g in zip(params, gradients)], step_loss

@jax.jit
def update_params_3(params, x1, x2, y, lr):
    gradients = grad(loss_3)(params, x1, x2, y)
    step_loss = loss_3(params, x1, x2, y)
    return [(p - lr * g) for p, g in zip(params, gradients)], step_loss

@jax.jit
def update_params_4(params, x1, x2, y, lr):
    gradients = grad(loss_4)(params, x1, x2, y)
    step_loss = loss_4(params, x1, x2, y)
    return [(p - lr * g) for p, g in zip(params, gradients)], step_loss

@jax.jit
def update_params_5(params, x1, x2, y, lr):
    gradients = grad(loss_5)(params, x1, x2, y)
    step_loss = loss_5(params, x1, x2, y)
    return [(p - lr * g) for p, g in zip(params, gradients)], step_loss


def decide_training(params, x1, x2, y, lr, stage):
    if stage == 1:
        params, step_loss = update_params_1(params, x1, x2, y, lr)
        return params, step_loss

    elif stage == 2:
        params, step_loss = update_params_2(params, x1, x2, y, lr)
        return params, step_loss

    elif stage == 3:
        params, step_loss = update_params_3(params, x1, x2, y, lr)
        return params, step_loss

    elif stage == 4:
        params, step_loss = update_params_4(params, x1, x2, y, lr)
        return params, step_loss

    elif stage == 5:
        params, step_loss = update_params_5(params, x1, x2, y, lr)
        return params, step_loss

    else:
        print('Bad index for the training stage.')
        return None
        
# Main function to train the network
def train_stages_neural_network(params, stage, level, lr=0.01, epochs=100):
    decimal_dataset = prepare_dataset(level, epochs)
    inputs_1, inputs_2, outputs_prev = transform_to_tridimensional_matrix(decimal_dataset)
    outputs = prepare_outputs(stage, inputs_1, inputs_2, outputs_prev)
    final_loss = 0
    # Train the network
    for epoch in range(epochs):
        # Update parameters at each step
        params, step_loss = decide_training(params, inputs_1[epoch], inputs_2[epoch], outputs[epoch], lr, stage)
        final_loss += step_loss

    final_loss = final_loss / epochs
    #print(f"Loss: {final_loss:.6f}")
    return params, final_loss


# Main function to test the network
def test_stages_neural_network(params, stage, visualize_errors=0, real_test=0):
    if real_test == 1:
        decimal_dataset = generate_test_dataset(n_max=12)
    else:
        decimal_dataset = generate_test_dataset()
    inputs_1, inputs_2, outputs_prev = transform_to_tridimensional_matrix(decimal_dataset)
    outputs = prepare_outputs(stage, inputs_1, inputs_2, outputs_prev)
    correct_predictions_count = 0
    test_size = inputs_1.shape[0]

    for i in range(test_size):
        prediction, binary_pred = predict(params, inputs_1[i], inputs_2[i], stage)
        #print(f'Se ha predicho {prediction} y era {outputs[i]}')
        if jnp.all(prediction == outputs[i]):  # Check if the prediction matches the expected output
            correct_predictions_count += 1  # Increment correct prediction count
        elif visualize_errors == 1:
            print(f'Ha fallado {decimal_dataset.iloc[i,0]} por {decimal_dataset.iloc[i,1]}.')
    return test_size, correct_predictions_count

# Predict using the trained neural network
def predict(params, x1, x2, stage):
    if stage == 1:
        binary_pred = neural_network_1(params, x1, x2)
        rounded_pred = np.round(binary_pred)
        return rounded_pred, binary_pred
    elif stage == 2:
        binary_pred = neural_network_2(params, x1, x2)
        rounded_pred = np.round(binary_pred)
        return rounded_pred, binary_pred
    elif stage == 3:
        binary_pred = neural_network_3(params, x1, x2)
        rounded_pred = np.round(binary_pred)
        return rounded_pred, binary_pred
    elif stage == 4:
        binary_pred = neural_network_4(params, x1, x2)
        rounded_pred = np.round(binary_pred)
        return rounded_pred, binary_pred
    elif stage == 5:
        binary_pred = neural_network_5(params, x1, x2)
        rounded_pred = np.round(binary_pred)
        return rounded_pred, binary_pred
    else:
        print('Bad index for the training stage.')
        return None


# Check if two models are equal
def are_models_equal(model_1, model_2):
    # Check that both lists have the same length
    if len(model_1) != len(model_2):
        return False

    # Compare each element in both lists
    for elem1, elem2 in zip(model_1, model_2):
        if isinstance(elem1, jnp.ndarray) and isinstance(elem2, jnp.ndarray):
            # Compare two JAX arrays
            if not jnp.all(jnp.isclose(elem1, elem2, atol=1e-2)):
                return False
        else:
            return False

    return True
	
def generate_model(epsilon_non_zeros = 0.01, epsilon_zeros = 0):
    W1_almost_perfect = np.zeros((8, 16)) + epsilon_zeros * np.random.randint(-10, 10)
    for k in range(4):
        for i in range(4):
            W1_almost_perfect[i, 12+i-k*4] = 1/2 + epsilon_non_zeros * np.random.randint(-10, 10)
            W1_almost_perfect[4+i, 4*i:4*i+4] = 1/2 + epsilon_non_zeros * np.random.randint(-10, 10)
    
    W2_almost_perfect = np.zeros((16, 28)) + epsilon_zeros * np.random.randint(-10, 10)
    for i in range(4):
        for j in range(4):
            W2_almost_perfect[i*4 + j, i*8 + j] = 1 + epsilon_non_zeros * np.random.randint(-10, 10)
    
    R3_almost_perfect = np.zeros((28)) + epsilon_zeros * np.random.randint(-10, 10)
    R4_almost_perfect = np.zeros((28)) + epsilon_zeros * np.random.randint(-10, 10)
    R5_almost_perfect = np.zeros((28)) + epsilon_zeros * np.random.randint(-10, 10)
    R6_almost_perfect = np.zeros((28)) + epsilon_zeros * np.random.randint(-10, 10)
    R7_almost_perfect = np.zeros((28)) + epsilon_zeros * np.random.randint(-10, 10)
    
    for i in range(4):
        R3_almost_perfect[7*i + 5] = 1 + epsilon_non_zeros * np.random.randint(-10, 10)
        R4_almost_perfect[7*i + 4] = 1 + epsilon_non_zeros * np.random.randint(-10, 10)
        R5_almost_perfect[7*i + 3] = 1 + epsilon_non_zeros * np.random.randint(-10, 10)
        R6_almost_perfect[7*i + 2] = 1 + epsilon_non_zeros * np.random.randint(-10, 10)
        R7_almost_perfect[7*i + 1] = 1 + epsilon_non_zeros * np.random.randint(-10, 10)
    
    v3_almost_perfect = 1/2 + epsilon_non_zeros * np.random.randint(-10, 10)
    v4_almost_perfect = 1/2 + epsilon_non_zeros * np.random.randint(-10, 10)
    v5_almost_perfect = 1/2 + epsilon_non_zeros * np.random.randint(-10, 10)
    v6_almost_perfect = 1/2 + epsilon_non_zeros * np.random.randint(-10, 10)
    
    T_almost_perfect = np.zeros((28,7)) + epsilon_zeros * np.random.randint(-10, 10)
    
    for i in range(7):
        for j in range(4):
            T_almost_perfect[7*j + i, i] = 1 + epsilon_non_zeros * np.random.randint(-10, 10)
    
    v7_almost_perfect = 1/2 + epsilon_non_zeros * np.random.randint(-10, 10)
    
    original_model = [W1_almost_perfect, 
                      W2_almost_perfect, 
                      R3_almost_perfect, R4_almost_perfect, R5_almost_perfect, R6_almost_perfect, R7_almost_perfect, 
                      v3_almost_perfect, v4_almost_perfect, v5_almost_perfect, v6_almost_perfect,
                      T_almost_perfect, v7_almost_perfect]
    trainable_model = [W1_almost_perfect, 
                       W2_almost_perfect, 
                       R3_almost_perfect, R4_almost_perfect, R5_almost_perfect, R6_almost_perfect, R7_almost_perfect, 
                       v3_almost_perfect, v4_almost_perfect, v5_almost_perfect, v6_almost_perfect,
                       T_almost_perfect, v7_almost_perfect]
    return trainable_model, original_model

class Tee(object):
    def __init__(self, file, mode='w'):
        self.file = open(file, mode)
        self.console = sys.stdout  

    def write(self, data):
        self.console.write(data)   
        self.file.write(data)    

    def flush(self):
        self.console.flush()
        self.file.flush()

    def close(self):
        self.file.close()

def load_trainable_model(model, stage, date, epsilon_zeros = 0, epsilon_non_zeros = 0):
    model_path = f'Structured_Network/Stages/Model_almost_perfect_zeros_{epsilon_zeros}_non-zeros_{epsilon_non_zeros}/Stage_{stage}/{model}_{stage}-{date}.pkl'
    with open(model_path, 'rb') as f:
        globals()[f'{model}_{stage}'] = pickle.load(f)
    print(f'Model {model}_{stage} loaded successfully.')
    return globals()[f'{model}_{stage}']
	

iters = 20
epsilon_non_zeros = 0.1
epsilon_zeros = 0.1

for iteration in range(iters):
    trainable_model, original_model = generate_model(epsilon_non_zeros = epsilon_non_zeros, epsilon_zeros = epsilon_zeros)
    h = 0.001 # Additional parameter needed for the two first stages
    trainable_model_stage_1 = [trainable_model[0], h]
    trainable_model_stage_2 = [trainable_model[1], h]
    trainable_model_stage_3 = [trainable_model[2], trainable_model[3], trainable_model[4], trainable_model[5], trainable_model[6], trainable_model[7], trainable_model[8], trainable_model[9], trainable_model[10]]
    trainable_model_stage_4 = [trainable_model[11], trainable_model[12]]
    training_stages = 5
    trainings_needed = np.zeros(training_stages)
    model = 'trainable_model_stage'
    folder_path = f"Parameters_AP/zeros_{epsilon_zeros}_non-zeros_{epsilon_non_zeros}"
    os.makedirs(folder_path, exist_ok=True) 
    
    current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    file_path = os.path.join(folder_path, f"{model}_{current_time}.pkl")
    
    with open(file_path, 'wb') as file:
        pickle.dump(original_model, file)
    
    print(f'Saved file: {model}_{current_time}.pkl')

    visualizer = 10
    lr_changer = 250
    stage_changer = 2000
    N = 500

    for stage in range(1,5): 
        save_dir = f"Structured_Network/Stages/Model_almost_perfect_zeros_{epsilon_zeros}_non-zeros_{epsilon_non_zeros}/Stage_{stage}" 
        os.makedirs(save_dir, exist_ok=True) 
        results_file = os.path.join(save_dir, f"Stage_{stage}_results_{current_time}.txt") 
        tee = Tee(results_file, 'w') 
        sys.stdout = tee       
    
        try:
            test_size, correct_predictions_count = test_stages_neural_network(params=globals()[f"{model}_{stage}"], stage=stage)
            final_loss = 1
            lr = 0.001
            level = -1
            response = "yes"
            while final_loss != 0:
                prev_model = globals()[f"{model}_{stage}"]
                globals()[f"{model}_{stage}"], final_loss = train_stages_neural_network(params=globals()[f"{model}_{stage}"], stage=stage, level=level, lr=lr, epochs=N)
                trainings_needed[stage-1] += 1
                
                if math.isnan(final_loss):
                    globals()[f"{model}_{stage}"] = prev_model
                    print('Loss is NaN')
                    break    

                if trainings_needed[stage-1] % visualizer == 0:
                    if response.lower() == "yes":
                        test_size, correct_predictions_count = test_stages_neural_network(params=globals()[f"{model}_{stage}"], stage=stage)
                        print(f"STAGE {stage}: Out of {test_size}, {correct_predictions_count} were predicted correctly in the current model.")
                    elif response.lower() == "no":
                        print(f"STAGE {stage}: Objective completed, all are predicted correctly in the current model.")
                    print(f"STAGE {stage}: Loss is {final_loss}.")
                
                #if trainings_needed[stage-1] % lr_changer == 0:
                #    new_lr = input(f"Change of learning rate? (Current one is {lr}, press enter if not): ")
                #    if new_lr.lower() == "no":
                #        pass
                #    elif new_lr != "":
                #        lr = float(new_lr)
                        
                if trainings_needed[stage-1] % stage_changer == 0:
                    response_pre = 'yes'
                    if response_pre.lower() == "yes":
                        break
					
                if correct_predictions_count == test_size:
                    response = "yes"
                    while response.lower() not in ["yes", "no"]:
                        response = input("Objective completed, skip to next stage? (yes/no): ")
                        if response.lower() not in ["yes", "no"]:
                            print('Incorrect answer')        
                    if response.lower() == "yes":
                        print('Objective completed')
                        break
                    elif response.lower() == "no":
                        test_size = correct_predictions_count + 1

            print(f'Stage {stage} completed in {trainings_needed[stage-1]} trainings.')
            save_response = 'yes'
            if save_response.lower() == 'yes':
                save_path = os.path.join(save_dir, f"{model}_{stage}-{current_time}.pkl")
                with open(save_path, 'wb') as f:
                    pickle.dump(globals()[f"{model}_{stage}"], f)
                print(f"Model {model}_{stage} saved at {save_path}")
               
        finally:
            sys.stdout = tee.console
            tee.close()
        print(f"Results of Stage {stage} saved in {results_file}")