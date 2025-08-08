import numpy as np
import os
import re
import random
import json
import sys
import time
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

epsilon = 0.3

param_type = 'AP'

folder = '/home/samuel_lozano/Curriculum_Learning/Easy_Multidigit_Addition_Decimal/'

# Cargar los módulos preentrenados (unit_module y carry_module)
unit_addition_model = load_model(f'{folder}Modules/unit_addition_module.keras')
unit_carry_model = load_model(f'{folder}Modules/unit_carry_module.keras')

unit_addition_model.trainable = False
unit_carry_model.trainable = False

unit_addition_model.name = 'unit_addition_model'
unit_carry_model.name = 'unit_carry_model'

# Cargar las parejas desde el archivo
with open(f"{folder}train_couples_stimuli.txt", "r") as file:
    train_couples = eval(file.read())

with open(f"{folder}stimuli.txt", "r") as file:
    test_couples = eval(file.read())

with open(f"{folder}test_dataset.txt", "r") as file:
    test_dataset = eval(file.read())

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
    
    # Calcular frecuencias basadas en exp(-(a+b)/N) y normalizar
    probabilities = np.array([np.exp(-(a + b) / 100) for a, b in train_couples])
    probabilities /= probabilities.sum()  # Normalizar para convertir en una distribución de probabilidad

    # Seleccionar parejas aleatoriamente según las probabilidades
    selected_indices = np.random.choice(len(train_couples), size=size_epoch, p=probabilities)
    selected_couples = [train_couples[i] for i in selected_indices]

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

# Función de pérdida
def loss_fn(params, x, y):
    y_pred_1, y_pred_2 = model(params, x)
    return jnp.mean((y_pred_1 - y[:, 0]) ** 2) + jnp.mean((y_pred_2 - y[:, 1]) ** 2)
    
# Función para actualizar los parámetros
def update_params(params, x, y, lr):
    # Asegúrate de usar JAX para los gradientes y operaciones
    gradients = grad(loss_fn)(params, x, y)
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


# Modelo dinámico en JAX
def construct_matrices(x):
    carry_unit_val = {}
    unit_val = {}

    for i in [0,1]:
       for j in [2,3]:
            units_inputs = jnp.array(x[:, [i, j]])
            units_input = units_inputs[:, None, :]
           
            unit_output = jnp.array(unit_addition_model(units_input))
            unit_carry_output = jnp.array(unit_carry_model(units_input))
           
            unit_val[f'{i}_{j}'] = jnp.argmax(unit_output, axis=-1)
            carry_unit_val[f'{i}_{j}'] = jnp.argmax(unit_carry_output, axis=-1)

    return carry_unit_val, unit_val


def model(params, x):
    carry_unit_val, unit_val = construct_matrices(x)
    
    salida_1 = sum(params[f'v_0_1_{i}_{j}'] * carry_unit_val[f'{i}_{j}'] +
        params[f'v_1_1_{i}_{j}'] * unit_val[f'{i}_{j}']
        for i in [0,1] for j in [2,3]
        )

    # Salida 2
    salida_2 = sum(params[f'v_0_2_{i}_{j}'] * carry_unit_val[f'{i}_{j}'] +
        params[f'v_1_2_{i}_{j}'] * unit_val[f'{i}_{j}']
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
    serializable_params = {key: value.tolist() for key, value in params.items()}
    
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

save_dir = f"{folder}Results_models/{param_type}_{epsilon}"
save_model_dir = f"{folder}Trained_models/{param_type}_{epsilon}"
save_model_dir_2 = f"{folder}Super_trained_models/{param_type}_{epsilon}"
folder_path = f'{folder}Parameters/{param_type}_{epsilon}'
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

    trainable_model_jnp = {key: jnp.array(value) for key, value in trainable_model.items()}
    print(f'Loaded trainable_model_{current_time}.json')
    
    os.makedirs(save_dir, exist_ok=True) 
    results_file = os.path.join(save_dir, f"Results_{current_time}.txt") 
    tee = Tee(results_file, 'w') 
    sys.stdout = tee
    
    try: 
        new_params, average_loss = train_model(trainable_model_jnp, train_couples, size_epoch=1000, lr=0.01, epochs=100, stop=1, batch_size=0)
        pred_count, pred_count_test = correct_predictions(new_params)
        trained_model_filename = f"trained_model_{current_time}.json"
        save_trained_model(new_params, trained_model_filename, save_model_dir)
        print(f'Saved trained_model_{current_time}.json')

        if pred_count != 5051:
            new_params_2, average_loss_2 = train_model(new_params, train_couples, size_epoch=1000, lr=0.01, epochs=500, stop=0, batch_size=0)
            trained_model_filename_2 = f"super_trained_model_{current_time}.json"
            save_trained_model(new_params_2, trained_model_filename_2, save_model_dir_2)
            print(f'Saved super_trained_model_{current_time}.json')

    finally:
        sys.stdout = tee.console
        tee.close()