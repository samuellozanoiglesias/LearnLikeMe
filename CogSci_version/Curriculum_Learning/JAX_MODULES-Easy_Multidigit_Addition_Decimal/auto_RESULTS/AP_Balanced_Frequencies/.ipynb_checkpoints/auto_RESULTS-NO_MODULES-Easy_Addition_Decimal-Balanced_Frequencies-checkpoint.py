import numpy as np
import os
import re
import random
import json
import sys
import time
import pickle
from datetime import datetime
import pandas as pd
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
from tensorflow.keras.models import Model, load_model, clone_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense, Multiply, Add, Lambda, Concatenate, Reshape, Flatten
from tensorflow.keras.initializers import GlorotUniform, RandomUniform, Constant
from tensorflow.keras.callbacks import LambdaCallback
import jax
import jax.numpy as jnp
from jax import grad, jit
from flax import linen as nn
from flax.training import train_state
import optax

if len(sys.argv) != 2:
    print("How to use: python3 script.py <epsilon_value>")
    sys.exit(1)

try:
    epsilon_raw = sys.argv[1]
    if '.' not in epsilon_raw:
        epsilon = int(epsilon_raw) 
    else:
        epsilon = float(epsilon_raw)
except ValueError:
    print("Error: the value of epsilon must be a number.")
    sys.exit(1)

print(f"Value of epsilon: {epsilon}")

param_type = 'AP'

folder = '/home/samuel_lozano/Curriculum_Learning/Easy_Multidigit_Addition_Decimal/'
folder_specific = '/home/samuel_lozano/Curriculum_Learning/Easy_Multidigit_Addition_Decimal/v1-Balanced_Frequencies/NO_MODULES/'

# Cargar las parejas desde el archivo
with open(f"{folder}train_couples_stimuli.txt", "r") as file:
    train_couples = eval(file.read())

with open(f"{folder}stimuli.txt", "r") as file:
    test_couples = eval(file.read())

with open(f"{folder}test_dataset.txt", "r") as file:
    test_dataset = eval(file.read())

small_problem_size = [pair for pair in train_couples if (pair[0] + pair[1]) < 40]
medium_problem_size = [pair for pair in train_couples if 40 <= (pair[0] + pair[1]) <= 60]
large_problem_size = [pair for pair in train_couples if (pair[0] + pair[1]) > 60]

def generate_test_dataset():
    x_data = []
    y_data = []
    
    for a, b in test_dataset:
        a_dec = a // 10  # Decena del primer número
        a_unit = a % 10  # Unidad del primer número
        b_dec = b // 10  # Decena del segundo número
        b_unit = b % 10  # Unidad del segundo número

        x_data.append([a_dec, a_unit, b_dec, b_unit])  # Entrada

        sum_units = (a_unit + b_unit) % 10
        carry_units = 1 if (a_unit + b_unit) >= 10 else 0
        sum_dec = (a_dec + b_dec + carry_units) % 10
        y_data.append([sum_dec, sum_units])  # Salida
    
    return jnp.array(x_data), jnp.array(y_data)

# Función para generar el dataset de entrenamiento dinámicamente
def generate_train_dataset(train_couples, size_epoch):
    x_data = []
    y_data = []
    
    # Calcular el número de muestras a seleccionar de cada clase
    total_classes = 3
    balanced_class_count = size_epoch // total_classes
    remaining = size_epoch - total_classes * balanced_class_count

    # Seleccionar las muestras balanceadas de cada clase
    balanced_small_indices = np.random.choice(len(small_problem_size), size=balanced_class_count, replace=True)
    balanced_small = [small_problem_size[i] for i in balanced_small_indices]
    balanced_medium_indices = np.random.choice(len(medium_problem_size), size=balanced_class_count, replace=True)
    balanced_medium = [medium_problem_size[i] for i in balanced_medium_indices]
    balanced_large_indices = np.random.choice(len(large_problem_size), size=balanced_class_count, replace=True)
    balanced_large = [large_problem_size[i] for i in balanced_large_indices]

    # Rellenar aleatoriamente las clases restantes
    remaining_classes = ['small', 'medium', 'large']
    remaining_classes_idx = np.random.choice(remaining_classes, size=remaining, replace=True)

    # Rellenar las clases con los elementos restantes
    for class_type in remaining_classes_idx:
        #print(class_type)
        if class_type == 'small':
            balanced_small_choice = np.random.choice(len(small_problem_size))
            balanced_small.append(small_problem_size[balanced_small_choice])
        elif class_type == 'medium':
            balanced_medium_choice = np.random.choice(len(medium_problem_size))
            balanced_medium.append(medium_problem_size[balanced_medium_choice])
        else:
            balanced_large_choice = np.random.choice(len(large_problem_size))
            balanced_large.append(large_problem_size[balanced_large_choice])

    # Seleccionar las parejas balanceadas
    #print(balanced_small)
    #print(balanced_medium)
    #print(balanced_large)
    selected_couples = np.concatenate((balanced_small, balanced_medium, balanced_large))
    np.random.shuffle(selected_couples)

    for a, b in selected_couples:
        a_dec = a // 10  # Decena del primer número
        a_unit = a % 10  # Unidad del primer número
        b_dec = b // 10  # Decena del segundo número
        b_unit = b % 10  # Unidad del segundo número

        x_data.append([a_dec, a_unit, b_dec, b_unit])  # Entrada

        sum_units = (a_unit + b_unit) % 10
        carry_units = 1 if (a_unit + b_unit) >= 10 else 0
        sum_dec = (a_dec + b_dec + carry_units) % 10
        y_data.append([sum_dec, sum_units])  # Salida
    
    return jnp.array(x_data), jnp.array(y_data)

# Modulos

# Clase que define la estructura del modelo
class carry_LSTMModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        lstm_1 = nn.LSTMCell(features=16)
        dense = nn.Dense(2)

        carry1 = lstm_1.initialize_carry(jax.random.PRNGKey(0), (x.shape[0],)) 

        for t in range(x.shape[1]):  # Iterar sobre los pasos temporales
            carry1, x_t = lstm_1(carry1, x[:, t])

        hidden_state = carry1[0] 
        final_output = nn.softmax(dense(hidden_state))
        return final_output

class unit_LSTMModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        lstm_1 = nn.LSTMCell(features=16)
        lstm_2 = nn.LSTMCell(features=32)
        lstm_3 = nn.LSTMCell(features=16)
        dense = nn.Dense(10)

        carry1 = lstm_1.initialize_carry(jax.random.PRNGKey(0), (x.shape[0],))
        carry2 = lstm_2.initialize_carry(jax.random.PRNGKey(1), (x.shape[0],))
        carry3 = lstm_3.initialize_carry(jax.random.PRNGKey(2), (x.shape[0],))

        for t in range(x.shape[1]):  # Iterar sobre los pasos temporales
            carry1, x_t = lstm_1(carry1, x[:, t])
            carry2, x_t = lstm_2(carry2, x_t)
            carry3, x_t = lstm_3(carry3, x_t)

        hidden_state = carry3[0]  # Estado oculto tiene forma (batch_size, 32)
        final_output = nn.softmax(dense(hidden_state))
        return final_output


# Crear el estado inicial del modelo cargado (sin entrenar)
def load_train_state(rng, learning_rate, initial_params, model):
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=initial_params, tx=tx)


# Función de pérdida
def loss_fn(params, x, y):
    y_pred_1, y_pred_2 = model(params, x)
    return jnp.mean((y_pred_1 - y[:, 0]) ** 2) + jnp.mean((y_pred_2 - y[:, 1]) ** 2)
    
# Función para actualizar los parámetros
def update_params(params, x, y, lr):
    # Asegúrate de usar JAX para los gradientes y operaciones
    gradients = grad(loss_fn)(params, x, y)
    #for key, gradient in gradients.items():
    #    print(f"Gradiente para {key}: {gradient}")
    new_params = jax.tree_util.tree_map(lambda p, g: p - lr * g, params, gradients)
    return new_params

# Función para entrenar el modelo
def train_model(params, train_couples, size_epoch, lr=0.01, epochs=100, stop=1, batch_size=0):
    final_loss = 0

    if batch_size == 0:
        batch_size = size_epoch

    # Entrenar el modelo
    for epoch in range(epochs): 
        x_train, y_train = generate_train_dataset(train_couples, size_epoch)
        total_examples = x_train.shape[0]

        batches_per_epoch = int(total_examples / batch_size)
        
        if epoch == 0:
            pred_count, pred_count_test, step_loss = correct_predictions_and_loss(params)  # Contamos las predicciones correctas
            print(f"Epoch {epoch}, Loss: {step_loss}, Correct predictions: {pred_count}, Correct predictions test: {pred_count_test}")
            
        for batch in range(batches_per_epoch):         
            x_batch = x_train[(batch * batch_size):((batch + 1) * batch_size)]
            y_batch = y_train[(batch * batch_size):((batch + 1) * batch_size)]
            params = update_params(params, x_batch, y_batch, lr)
            
        # Mostrar estadísticas cada 10 épocas
        if epoch % 5 == 0 and epoch != 0:
            pred_count, pred_count_test, step_loss = correct_predictions_and_loss(params)  # Contamos las predicciones correctas
            print(f"Epoch {epoch}, Loss: {step_loss}, Correct predictions: {pred_count}, Correct predictions test: {pred_count_test}")
            
            # Criterios de parada
            if stop == 1 and pred_count_test == 192:
                break
            if stop == 0 and pred_count == 5050:
                break
            if step_loss >= 1000000:
                break
        
    return params, step_loss

def correct_predictions_and_loss(params):
    x_test, y_test = generate_test_dataset()
    pred_count = 0
    pred_count_test = 0
    total_examples = x_test.shape[0]
    pred_tens, pred_units = model(params, x_test)   
    loss = jnp.mean((pred_tens - y_test[:, 0]) ** 2) + jnp.mean((pred_units - y_test[:, 1]) ** 2)
    for i in range(total_examples):
        normalized_pred = [int(jnp.round(pred_tens[i].item())),
                           int(jnp.round(pred_units[i].item()))]
        
        # Obtener los valores a y b de x_test
        a = int(str(x_test[i, 0]) + str(x_test[i, 1]))
        b = int(str(x_test[i, 2]) + str(x_test[i, 3]))
        # Comparar las predicciones con las etiquetas y contar los aciertos
        if normalized_pred[0] == y_test[i, 0] and normalized_pred[1] == y_test[i, 1]:
            pred_count += 1
            if (a, b) in test_couples:
                pred_count_test += 1

    return pred_count, pred_count_test, loss

def correct_predictions(params):
    x_test, y_test = generate_test_dataset()
    pred_count = 0
    pred_count_test = 0
    total_examples = x_test.shape[0]
    pred_tens, pred_units = model(params, x_test)        
    for i in range(total_examples):
        normalized_pred = [int(jnp.round(pred_tens[i].item())),
                           int(jnp.round(pred_units[i].item()))]
        
        # Obtener los valores a y b de x_test
        a = int(str(x_test[i, 0]) + str(x_test[i, 1]))
        b = int(str(x_test[i, 2]) + str(x_test[i, 3]))
        # Comparar las predicciones con las etiquetas y contar los aciertos
        if normalized_pred[0] == y_test[i, 0] and normalized_pred[1] == y_test[i, 1]:
            pred_count += 1
            if (a, b) in test_couples:
                pred_count_test += 1

    return pred_count, pred_count_test


def smooth_argmax_bivariate(x, k=10):
    y = 1 / (1 + jnp.exp(-k * (x[:, 1] - x[:, 0])))
    return y

def smooth_argmax_multivariate(x):
    smoothed_index = jnp.sum(x * jnp.arange(x.shape[1]), axis=-1)
    return smoothed_index

# Modelo dinámico en JAX
def model(params, x):
    carry_val = {}
    unit_val = {}
    decision_module_params = params['decision_params']
    unit_module_params = params['unit_params'] 
    carry_module_params = params['carry_params'] 
    
    for i in [0,1]:
        for j in [2,3]:
            units_inputs = jnp.array(x[:, [i, j]])
            units_input = units_inputs[:, None, :]
           
            unit_output = unit_LSTMModel().apply({'params': unit_module_params}, units_input)
            carry_output = carry_LSTMModel().apply({'params': carry_module_params}, units_input)
           
            unit_val[f'{i}_{j}'] = smooth_argmax_multivariate(unit_output)
            carry_val[f'{i}_{j}'] = smooth_argmax_bivariate(carry_output)
            #print(unit_val[f'0_2'])
            #print(carry_val[f'0_2'])
            
    salida_1 = sum(decision_module_params[f'v_0_1_{i}_{j}'] * carry_val[f'{i}_{j}'] +
        decision_module_params[f'v_1_1_{i}_{j}'] * unit_val[f'{i}_{j}']
        for i in [0,1] for j in [2,3]
        )

    # Salida 2
    salida_2 = sum(decision_module_params[f'v_0_2_{i}_{j}'] * carry_val[f'{i}_{j}'] +
        decision_module_params[f'v_1_2_{i}_{j}'] * unit_val[f'{i}_{j}']
        for i in [0,1] for j in [2,3]
        )

    return salida_1, salida_2

# Función para leer parámetros de un archivo JSON
def load_params_from_file(filename):
    with open(filename, 'r') as f:
        return json.load(f)

# Guardar el modelo entrenado
def save_trained_model(params, filename, model_dir):
    os.makedirs(model_dir, exist_ok=True)
    file_path = os.path.join(model_dir, filename)
    serializable_params = convert_to_serializable(params)
    
    with open(file_path, 'w') as f:
        json.dump(serializable_params, f)

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

def convert_to_jnp_arrays(data):
    """Convierte listas y números en un diccionario en arreglos de JAX recursivamente."""
    if isinstance(data, dict):
        # Si el valor es un diccionario, aplica la conversión de forma recursiva
        return {key: convert_to_jnp_arrays(value) for key, value in data.items()}
    elif isinstance(data, list):
        # Si el valor es una lista, conviértela en un arreglo de JAX
        return jnp.array(data)
    elif isinstance(data, (int, float)):
        # Si el valor es un número, conviértelo directamente en un arreglo de JAX
        return jnp.array(data)
    else:
        # Deja otros tipos (como cadenas) sin cambios
        return data

def convert_to_serializable(data):
    """Convierte arreglos de JAX y listas en datos serializables recursivamente."""
    if isinstance(data, dict):
        # Si el valor es un diccionario, aplica la conversión recursiva
        return {key: convert_to_serializable(value) for key, value in data.items()}
    elif isinstance(data, jnp.ndarray):
        # Convierte arreglos de JAX en listas
        return data.tolist()
    elif isinstance(data, list):
        # Convierte listas en datos serializables
        return [convert_to_serializable(item) for item in data]
    else:
        # Deja otros tipos (como números o cadenas) sin cambios
        return data

save_dir = f"{folder_specific}Results_models/{param_type}_{epsilon}"
save_model_dir = f"{folder_specific}Trained_models/{param_type}_{epsilon}"
save_model_dir_2 = f"{folder_specific}Super_trained_models/{param_type}_{epsilon}"
folder_path = f'{folder_specific}Parameters/{param_type}_{epsilon}'
date_pattern = r'trainable_model_(\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}).json'
files = sorted(
    (f for f in os.listdir(folder_path) if not f.startswith('.')),  # Filtrar archivos ocultos
    key=lambda x: re.search(date_pattern, x).group(1) if re.search(date_pattern, x) else ''
)

for filename in files:
    match = re.search(date_pattern, filename)
    if match:
        current_time = match.group(1)
    else:
        print('Error')
        break
    
    file_path = f"{folder_path}/trainable_model_{current_time}.json"
    with open(file_path, 'rb') as file:
        trainable_model = json.load(file)

    trainable_model_jnp = convert_to_jnp_arrays(trainable_model)
    print(f'Loaded trainable_model_{current_time}.json')
    
    os.makedirs(save_dir, exist_ok=True) 
    results_file = os.path.join(save_dir, f"Results_{current_time}.txt") 
    tee = Tee(results_file, 'w') 
    sys.stdout = tee
    
    try: 
        new_params, average_loss = train_model(trainable_model_jnp, train_couples, size_epoch=1000, lr=0.01, epochs=1000, stop=1, batch_size=0)
        pred_count, pred_count_test = correct_predictions(new_params)
        trained_model_filename = f"trained_model_{current_time}.json"
        save_trained_model(new_params, trained_model_filename, save_model_dir)
        print(f'Saved trained_model_{current_time}.json')    
    finally:
        sys.stdout = tee.console
        tee.close()