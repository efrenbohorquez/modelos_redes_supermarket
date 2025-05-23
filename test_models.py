"""
Script para probar la carga de modelos TensorFlow.
Este script intenta cargar los modelos MLP, LSTM, CNN y Híbrido para verificar si el problema se ha resuelto.
"""

import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import sys
import numpy as np

# Definir métricas personalizadas para cargar los modelos
def mse(y_true, y_pred):
    return tf.keras.metrics.mean_squared_error(y_true, y_pred)

def mae(y_true, y_pred):
    return tf.keras.metrics.mean_absolute_error(y_true, y_pred)

# Registrar las métricas personalizadas con TensorFlow
tf.keras.utils.get_custom_objects().update({
    'mse': mse,
    'mae': mae
})

# Rutas de los modelos
model_paths = {
    'MLP': 'models/mlp/mlp_ventas_totales.h5',
    'LSTM': 'models/lstm/lstm_ventas_totales_seq7.h5',
    'CNN': 'models/cnn/cnn_ventas_totales.h5',
    'Híbrido': 'models/hybrid/hybrid_model.h5'
}

# Intentar cargar cada modelo
for name, path in model_paths.items():
    try:
        print(f"Intentando cargar modelo {name} desde {path}...")
        if os.path.exists(path):
            model = load_model(path)
            print(f"✅ Modelo {name} cargado correctamente.")
            # Imprimir información del modelo
            model.summary(line_length=80)
            print("\n" + "="*80 + "\n")
        else:
            print(f"❌ El archivo del modelo {name} no existe en la ruta especificada.")
    except Exception as e:
        print(f"❌ Error al cargar el modelo {name}: {e}")
        print("\n")
