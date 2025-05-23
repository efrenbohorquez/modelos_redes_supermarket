"""
Script para probar la carga de los modelos TensorFlow-based.
Este script carga cada modelo y verifica si se puede cargar correctamente.
"""

import os
import tensorflow as tf
import joblib
from tensorflow.keras.models import load_model

# Definir funciones personalizadas para las métricas
def mse(y_true, y_pred):
    return tf.keras.metrics.mean_squared_error(y_true, y_pred)

def mae(y_true, y_pred):
    return tf.keras.metrics.mean_absolute_error(y_true, y_pred)

# Registrar las métricas personalizadas con TensorFlow
tf.keras.utils.get_custom_objects().update({
    'mse': mse,
    'mae': mae
})

def test_model_loading():
    print("Probando la carga de modelos...")
    models = {}
    # Definir rutas de los modelos
    model_paths = {
        'MLP': 'models/mlp/mlp_ventas_totales.h5',
        'LSTM': 'models/lstm/lstm_ventas_totales_seq7.h5',
        'CNN': 'models/cnn/cnn_ventas_totales.h5',
        'Baseline': 'models/baseline/models/rf_optimized_model.joblib',
        'Híbrido': 'models/hybrid/hybrid_model.h5',
        'Ensemble': 'models/ensemble/stacking_regressor.joblib'
    }
    
    # Verificar si existen los modelos y cargarlos
    for name, path in model_paths.items():
        try:
            if os.path.exists(path):
                print(f"Verificando modelo {name}...")
                if path.endswith('.h5'):
                    # Cargar modelo TensorFlow/Keras
                    model = load_model(path)
                    print(f"✅ Modelo {name} cargado correctamente")
                    # Mostrar un resumen del modelo si es de Keras
                    if hasattr(model, 'summary'):
                        print(f"Resumen del modelo {name}:")
                        model.summary()
                else:
                    # Cargar modelo scikit-learn/joblib
                    model = joblib.load(path)
                    print(f"✅ Modelo {name} cargado correctamente")
                models[name] = model
            else:
                print(f"❌ No se encontró el archivo del modelo {name} en {path}")
        except Exception as e:
            print(f"❌ Error al cargar el modelo {name}: {str(e)}")
    
    return models

if __name__ == "__main__":
    models = test_model_loading()
    print("\nResumen de la prueba:")
    print(f"Total de modelos probados: {len(models)}")
    print(f"Modelos cargados correctamente: {list(models.keys())}")
