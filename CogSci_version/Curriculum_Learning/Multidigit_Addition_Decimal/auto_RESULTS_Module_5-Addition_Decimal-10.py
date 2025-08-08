import numpy as np
import os
import re
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
from jax import random, grad, jit

epsilon = 10

# Cargar los módulos preentrenados (unit_module y carry_module)
unit_addition_model = load_model('unit_addition_module.keras')
unit_carry_model = load_model('unit_carry_module.keras')
dec_addition_model = load_model('dec_addition_module.keras')
dec_carry_model = load_model('dec_carry_module.keras')

unit_addition_model.trainable = False
unit_carry_model.trainable = False
dec_addition_model.trainable = False
dec_carry_model.trainable = False

unit_addition_model.name = 'unit_addition_model'
unit_carry_model.name = 'unit_carry_model'
dec_addition_model.name = 'dec_addition_model'
dec_carry_model.name = 'dec_carry_model'


# Cargar las parejas desde el archivo
with open(f"train_couples.txt", "r") as file:
    train_couples = eval(file.read())

with open(f"test_dataset.txt", "r") as file:
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
        carry_dec = 1 if (a_dec + b_dec + carry_units) >= 10 else 0
        y_data.append([carry_dec, sum_dec, sum_units])  # Salida
    
    return jnp.array(x_data), jnp.array(y_data)


# Función para leer los datos desde un archivo .txt y generar el dataset de entrenamiento
def generate_train_dataset():
    x_data = []
    y_data = []
    
    for a, b in train_couples:
        a_dec = a // 10  # Decena del primer número
        a_unit = a % 10  # Unidad del primer número
        b_dec = b // 10  # Decena del segundo número
        b_unit = b % 10  # Unidad del segundo número

        x_data.append([a_dec, a_unit, b_dec, b_unit])  # Entrada

        sum_units = (a_unit + b_unit) % 10
        carry_units = 1 if (a_unit + b_unit) >= 10 else 0
        sum_dec = (a_dec + b_dec + carry_units) % 10
        carry_dec = 1 if (a_dec + b_dec + carry_units) >= 10 else 0
        y_data.append([carry_dec, sum_dec, sum_units])  # Salida
    
    return jnp.array(x_data), jnp.array(y_data)

# Función para generar los datos
def generate_final_data():
    x_data = []
    y_data = []
    for a_dec in range(10):
        for a_unit in range(10):
            for b_dec in range(10):
                for b_unit in range(10):
                    x_data.append([a_dec, a_unit, b_dec, b_unit])  # Entrada
                    sum_units = (a_unit + b_unit) % 10
                    carry_units = 1 if (a_unit + b_unit) >= 10 else 0
                    sum_dec = (a_dec + b_dec + carry_units) % 10
                    carry_dec = 1 if (a_dec + b_dec + carry_units) >= 10 else 0
                    y_data.append([carry_dec, sum_dec, sum_units])  # Salida
    return jnp.array(x_data), jnp.array(y_data)

# Función para crear parámetros entrenables (v_0, ..., v_11)
def init_params(epsilon = 0.1):
    v_values_init = jnp.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1], dtype=jnp.float32)
    key = random.PRNGKey(0)
    keys = random.split(key, 12)
    v_params = {f'v{i}': random.normal(keys[i], (1,)) * epsilon + v_values_init[i] for i in range(12)}
    return v_params

# Función de pérdida
def loss_fn(params, x, y):
    y_pred_1, y_pred_2, y_pred_3 = model(params, x)
    return jnp.mean((y_pred_1 - y[0]) ** 2) + jnp.mean((y_pred_2 - y[1]) ** 2) + jnp.mean((y_pred_3 - y[2]) ** 2)
    
# Función para entrenar el modelo
def update_params(params, x, y, lr):
    # Asegúrate de usar JAX para los gradientes y operaciones
    gradients = grad(loss_fn)(params, x, y)
    step_loss = loss_fn(params, x, y)
    new_params = jax.tree.map(lambda p, g: p - lr * g, params, gradients)
    return new_params, step_loss

def correct_predictions(params, x_train, y_train):
    pred_count = 0
    total_examples =  x_train.shape[0]   
    
    for i in range(total_examples):
        prediction = model(params, x_train[i])
        # Obtener las predicciones para la unidad, decena y acarreo
        normalized_pred = [int(jnp.round(prediction[j].item())) for j in range(3)]
        
        if normalized_pred[0] == y_train[i,0] and normalized_pred[1] == y_train[i,1] and normalized_pred[2] == y_train[i,2]:
            pred_count += 1
            
    return pred_count

def train_model(params, x_train, y_train, lr=0.01, epochs=100):
    final_loss = 0
    # Convertir x_train y y_train a arrays de JAX (si aún no lo son)
    x_train = jnp.array(x_train)
    y_train = jnp.array(y_train)
    
    # Entrenar el modelo
    for epoch in range(epochs):  # Número de épocas
        params, step_loss = update_params(params, x_train[epoch], y_train[epoch], lr)
        final_loss += step_loss
        if (epoch + 1) % 10 == 0:
            pred_count = correct_predictions(params, x_train, y_train)  # Contamos las predicciones correctas
            print(f"Epoch {epoch}, Loss: {step_loss}, Correct predictions: {pred_count}")

            if pred_count == 10000:
                break
        
    final_loss = final_loss / epochs
    return params, final_loss

# Función para imprimir las predicciones y el loss en cada época
def print_predictions_and_loss(epoch, predictions, y_train):
    pred_count = 0
    total_examples = x_train.shape[0]
    
    for i in range(total_examples):
        # Obtener las predicciones para la unidad, decena y acarreo
        normalized_pred = [int(jnp.round(predictions[j][i])) for j in range(3)]
        
        # Concatenar las predicciones en un número de 3 dígitos
        concatenated_pred = int("".join(str(pred) for pred in normalized_pred))
        
        # Generar la salida esperada, concatenando los valores reales de y_train
        expected_output = int("".join(str(int(round(val))) for val in y_train[i]))
        
        # Comprobar si la predicción es igual a la salida esperada
        if concatenated_pred == expected_output:
            pred_count += 1

    print(f"Epoch {epoch + 1}:")
    print(f"Predicciones correctas: {pred_count} de {total_examples}")
    print("-" * 40)

    # Si todas las predicciones son correctas, detener el entrenamiento
    if pred_count == total_examples:
        print("¡Todas las combinaciones han sido aprendidas correctamente! Deteniendo entrenamiento.")
        return True
    return False

def predictions(params, x_train, y_train):
    pred_count = 0
    total_examples = x_train.shape[0]   
    
    for i in range(total_examples):
        prediction = model(params, x_train[i])
        # Obtener las predicciones para la unidad, decena y acarreo
        normalized_pred = [int(jnp.round(prediction[j].item())) for j in range(3)]
        
        # Concatenar las predicciones en un número de 3 dígitos
        concatenated_pred = int("".join(str(pred) for pred in normalized_pred))
        
        # Generar la salida esperada, concatenando los valores reales de y_train
        expected_output = int("".join(str(int(round(val))) for val in y_train[i]))
        
        # Comprobar si la predicción es igual a la salida esperada
        if concatenated_pred == expected_output:
            pred_count += 1

    print(f"Predicciones correctas: {pred_count} de {total_examples}")

    if pred_count == total_examples:
        print("¡Todas las combinaciones han sido aprendidas correctamente!")

def count_predictions(params, x_train, y_train):
    pred_count = 0
    total_examples =  x_train.shape[0]   
    
    for i in range(total_examples):
        prediction = model(params, x_train[i])
        # Obtener las predicciones para la unidad, decena y acarreo
        normalized_pred = [int(jnp.round(prediction[j].item())) for j in range(3)]
        
        if normalized_pred[0] == y_train[i,0]:
            if normalized_pred[1] == y_train[i,1]:
                if normalized_pred[2] == y_train[i,2]:
                    pred_count += 1

        if (normalized_pred[0] != y_train[i,0]) or (normalized_pred[1]!= y_train[i,1]) or (normalized_pred[2] != y_train[i,2]):
            print(f'Error en la suma: {x_train[i]}')
            print(f'Predicción: {[normalized_pred[0], normalized_pred[1], normalized_pred[2]]}')
            break
            
    print(f"Predicciones correctas: {pred_count} de {total_examples}")

    if pred_count == total_examples:
        print("¡Todas las combinaciones han sido aprendidas correctamente!")
        

# Modelo dinámico en JAX
def model(params, x):
    # Extraer unidades y decenas de los valores de entrada
    units_input = jnp.array([x[1], x[3]])
    units_input = units_input[None, None, :]
    
    # Llamar a los modelos unit_module y carry_module
    unit_output = jnp.array(unit_addition_model(units_input))  # Salida para unidades
    unit_carry_output = jnp.array(unit_carry_model(units_input))  # Salida de acarreo de unidades

    # Tomar el valor máximo de las predicciones (argmax en JAX)
    unit_val = jnp.argmax(unit_output, axis=-1)
    carry_unit_val = jnp.argmax(unit_carry_output, axis=-1)
    
    decs_input = jnp.array([x[0], x[2], carry_unit_val[0]])
    decs_input = decs_input[None, None, :]
    
    dec_output = jnp.array(dec_addition_model(decs_input))  # Salida para decenas
    dec_carry_output = jnp.array(dec_carry_model(decs_input))  # Salida de acarreo de decenas
    
    dec_val = jnp.argmax(dec_output, axis=-1)
    carry_dec_val = jnp.argmax(dec_carry_output, axis=-1)

    # Calcular las salidas combinadas con los parámetros v
    salida_1 = (params['v0'] * carry_dec_val) + (params['v1'] * dec_val) + (params['v2'] * carry_unit_val) + (params['v3'] * unit_val)
    salida_2 = (params['v4'] * carry_dec_val) + (params['v5'] * dec_val) + (params['v6'] * carry_unit_val) + (params['v7'] * unit_val)
    salida_3 = (params['v8'] * carry_dec_val) + (params['v9'] * dec_val) + (params['v10'] * carry_unit_val) + (params['v11'] * unit_val)

    return salida_1, salida_2, salida_3

# Función para leer parámetros de un archivo JSON
def load_params_from_file(filename):
    with open(filename, 'r') as f:
        return json.load(f)

# Guardar el modelo entrenado
def save_trained_model(params, filename):
    with open(filename, 'w') as f:
        json.dump(params, f)

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
        
x_train, y_train = generate_train_dataset()

save_dir = f"Trained_models/AP_{epsilon}"
folder_path = f'Parameters/AP_{epsilon}'
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
    
    file_path = f"Parameters/AP_{epsilon}/trainable_model_{current_time}.json"
    with open(file_path, 'rb') as file:
        trainable_model = json.load(file)

    trainable_model_jnp = {key: jnp.array(value) for key, value in trainable_model.items()}
    print(f'Loaded trainable_model_{current_time}.json')
    
    os.makedirs(save_dir, exist_ok=True) 
    results_file = os.path.join(save_dir, f"Results_{current_time}.txt") 
    tee = Tee(results_file, 'w') 
    sys.stdout = tee
    
    try: 
        new_params, average_loss = train_model(trainable_model_jnp, x_train, y_train, lr=0.01, epochs=250)
    
        trained_model_filename = f"trained_model_{current_time}.json"
        save_trained_model(new_params, trained_model_filename)
        print(f'Saved trained_model_{current_time}.json')

    finally:
        sys.stdout = tee.console
        tee.close()