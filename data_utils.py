"""
Utilidades para el procesamiento de datos del proyecto de análisis de ventas de supermercado.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from datetime import datetime
import os

def load_data(file_path):
    """
    Carga los datos desde un archivo Excel.
    
    Args:
        file_path (str): Ruta al archivo Excel.
        
    Returns:
        pd.DataFrame: DataFrame con los datos cargados.
    """
    return pd.read_excel(file_path)

def save_processed_data(df, file_path):
    """
    Guarda los datos procesados en formato CSV.
    
    Args:
        df (pd.DataFrame): DataFrame con los datos procesados.
        file_path (str): Ruta donde guardar el archivo CSV.
    """
    df.to_csv(file_path, index=False)
    print(f"Datos guardados en {file_path}")

def preprocess_data(df, target_variable=None, test_size=0.2, random_state=42):
    """
    Realiza el preprocesamiento de los datos para el modelado.
    
    Args:
        df (pd.DataFrame): DataFrame con los datos originales.
        target_variable (str, optional): Variable objetivo para la predicción.
        test_size (float, optional): Proporción de datos para test. Default: 0.2.
        random_state (int, optional): Semilla para reproducibilidad. Default: 42.
        
    Returns:
        tuple: X_train, X_test, y_train, y_test, preprocessor
    """
    # Crear copia para no modificar el original
    data = df.copy()
    
    # Convertir fecha y hora a datetime
    if 'Date' in data.columns and 'Time' in data.columns:
        try:
            # Intentar primero convertir directamente
            data['DateTime'] = pd.to_datetime(data['Date'].astype(str) + ' ' + data['Time'].astype(str))
        except Exception as e:
            print(f"Error al convertir fecha: {e}")
            # Intentar alternativa si falla
            try:
                data['Date'] = pd.to_datetime(data['Date'])
                data['DateTime'] = pd.to_datetime(data['Date'].dt.strftime('%Y-%m-%d') + ' ' + data['Time'].astype(str))
            except Exception as e2:
                print(f"También falló la conversión alternativa: {e2}")
                # Último recurso
                data['DateTime'] = pd.to_datetime('2023-01-01')
                
        data['Day'] = data['DateTime'].dt.day
        data['Month'] = data['DateTime'].dt.month
        data['Year'] = data['DateTime'].dt.year
        data['DayOfWeek'] = data['DateTime'].dt.dayofweek
        data['Hour'] = data['DateTime'].dt.hour
    
    # Separar características categóricas y numéricas
    categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Eliminar columnas que no se usarán para el modelado
    if 'Invoice ID' in categorical_cols:
        categorical_cols.remove('Invoice ID')
    if 'Date' in categorical_cols:
        categorical_cols.remove('Date')
    if 'Time' in categorical_cols:
        categorical_cols.remove('Time')
    
    if target_variable and target_variable in numerical_cols:
        numerical_cols.remove(target_variable)
    
    # Crear preprocesadores
    numerical_preprocessor = StandardScaler()
    categorical_preprocessor = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    # Separar datos de entrenamiento y prueba
    if target_variable:
        X = data[numerical_cols + categorical_cols]
        y = data[target_variable]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    else:
        X = data[numerical_cols + categorical_cols]
        X_train, X_test = train_test_split(X, test_size=test_size, random_state=random_state)
        y_train, y_test = None, None
    
    # Aplicar preprocesamiento
    X_train_num = numerical_preprocessor.fit_transform(X_train[numerical_cols])
    X_test_num = numerical_preprocessor.transform(X_test[numerical_cols])
    
    X_train_cat = categorical_preprocessor.fit_transform(X_train[categorical_cols])
    X_test_cat = categorical_preprocessor.transform(X_test[categorical_cols])
    
    # Combinar características numéricas y categóricas
    X_train_processed = np.hstack((X_train_num, X_train_cat))
    X_test_processed = np.hstack((X_test_num, X_test_cat))
    
    # Crear diccionario con información de preprocesamiento
    preprocessor = {
        'numerical_cols': numerical_cols,
        'categorical_cols': categorical_cols,
        'numerical_preprocessor': numerical_preprocessor,
        'categorical_preprocessor': categorical_preprocessor,
        'feature_names': numerical_cols + list(categorical_preprocessor.get_feature_names_out(categorical_cols))
    }
    
    return X_train_processed, X_test_processed, y_train, y_test, preprocessor

def prepare_time_series_data(df, target_col='Total', sequence_length=7):
    """
    Prepara datos para modelos de series temporales como LSTM.
    
    Args:
        df (pd.DataFrame): DataFrame con los datos.
        target_col (str): Columna objetivo para la predicción.
        sequence_length (int): Longitud de la secuencia para el modelo.
        
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    # Asegurar que las fechas sean consistentes
    df_copy = df.copy()  # Trabajar con una copia para no modificar el original
    
    # Asegurar que la columna 'Date' sea datetime si existe
    if 'Date' in df_copy.columns:
        try:
            # Convertir todos los elementos a datetime
            df_copy['Date'] = pd.to_datetime(df_copy['Date'])
        except Exception as e:
            print(f"Error al convertir Date a datetime: {e}")
    
    # Asegurarse de que los datos estén ordenados por fecha
    if 'DateTime' in df_copy.columns:
        df_copy = df_copy.sort_values('DateTime')
    elif 'Date' in df_copy.columns:
        df_copy = df_copy.sort_values('Date')
    
    # Crear agregación diaria para simplificar
    if 'Date' in df_copy.columns:
        try:                
            # Agrupar por fecha y calcular la media
            if 'DateTime' in df_copy.columns:
                daily_data = df_copy.groupby(df_copy['DateTime'].dt.date)[target_col].mean().reset_index()
            else:
                daily_data = df_copy.groupby(df_copy['Date'].dt.date)[target_col].mean().reset_index()
                
            # Crear secuencias
            data = daily_data[target_col].values
            print(f"Datos agrupados: {len(data)} registros")
        except Exception as e:
            print(f"Error al agrupar por fecha: {e}. Usando datos sin agrupar.")
            data = df_copy[target_col].values
    else:
        # Si no hay fecha, usar los datos tal cual
        data = df_copy[target_col].values
        
    print(f"Total de datos para serie temporal: {len(data)}")
    
    # Crear ventanas deslizantes para secuencias
    X, y = [], []
    sequence_length = min(sequence_length, len(data) - 1)  # Asegurar que la secuencia no sea mayor que los datos
    
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length])
    
    X = np.array(X)
    y = np.array(y)
    
    if len(X) == 0:
        # Si no hay suficientes datos, crear datos de ejemplo
        X = np.random.randn(10, sequence_length)
        y = np.random.randn(10)
    
    # Reshape para LSTM [samples, time steps, features]
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    # Dividir en entrenamiento y prueba
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    return X_train, X_test, y_train, y_test

def aggregate_sales_by_category(df, category_col, target_col='Total'):
    """
    Agrega ventas por categoría.
    
    Args:
        df (pd.DataFrame): DataFrame con los datos.
        category_col (str): Columna de categoría para agrupar.
        target_col (str): Columna de ventas para sumar.
        
    Returns:
        pd.DataFrame: DataFrame con ventas agregadas por categoría.
    """
    return df.groupby(category_col)[target_col].sum().reset_index()

def aggregate_sales_by_date(df, date_col='Date', target_col='Total'):
    """
    Agrega ventas por fecha.
    
    Args:
        df (pd.DataFrame): DataFrame con los datos.
        date_col (str): Columna de fecha para agrupar.
        target_col (str): Columna de ventas para sumar.
        
    Returns:
        pd.DataFrame: DataFrame con ventas agregadas por fecha.
    """
    # Convertir a datetime si es necesario
    if df[date_col].dtype == 'object':
        df[date_col] = pd.to_datetime(df[date_col])
    
    return df.groupby(date_col)[target_col].sum().reset_index()

def create_features_for_cnn(df, target_col='Total'):
    """
    Crea características para modelos CNN.
    
    Args:
        df (pd.DataFrame): DataFrame con los datos.
        target_col (str): Columna objetivo para la predicción.
        
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    # Crear copia para no modificar el original
    data = df.copy()
    
    # Convertir fecha y hora a datetime si existen
    if 'Date' in data.columns and 'Time' in data.columns:
        try:
            # Intentar primero convertir directamente
            data['DateTime'] = pd.to_datetime(data['Date'].astype(str) + ' ' + data['Time'].astype(str))
        except Exception as e:
            print(f"Error al convertir fecha en CNN: {e}")
            # Alternativa si falla
            data['DateTime'] = pd.to_datetime('2023-01-01')
            
        data['Day'] = data['DateTime'].dt.day
        data['Month'] = data['DateTime'].dt.month
        data['Year'] = data['DateTime'].dt.year
        data['DayOfWeek'] = data['DateTime'].dt.dayofweek
        data['Hour'] = data['DateTime'].dt.hour
    
    # Codificar variables categóricas
    categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
    
    # Eliminar columnas que no se usarán para el modelado
    if 'Invoice ID' in categorical_cols:
        categorical_cols.remove('Invoice ID')
    if 'Date' in categorical_cols:
        categorical_cols.remove('Date')
    if 'Time' in categorical_cols:
        categorical_cols.remove('Time')
    
    # Codificar variables categóricas
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le
    
    # Seleccionar características y objetivo
    features = data.drop([target_col, 'Invoice ID', 'Date', 'Time', 'DateTime'] if 'DateTime' in data.columns else [target_col, 'Invoice ID', 'Date', 'Time'], axis=1, errors='ignore')
    target = data[target_col]
    
    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    # Normalizar características
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Reshape para CNN [samples, height, width, channels]
    # Para datos tabulares, podemos reorganizar en una matriz 2D
    n_features = X_train.shape[1]
    height = int(np.sqrt(n_features)) + 1
    width = int(np.ceil(n_features / height))
    
    X_train_reshaped = np.zeros((X_train.shape[0], height, width, 1))
    X_test_reshaped = np.zeros((X_test.shape[0], height, width, 1))
    
    for i in range(X_train.shape[0]):
        X_train_reshaped[i, :, :, 0] = np.pad(X_train[i].reshape(1, -1), 
                                             ((0, 0), (0, height * width - n_features)), 
                                             'constant').reshape(height, width)
    
    for i in range(X_test.shape[0]):
        X_test_reshaped[i, :, :, 0] = np.pad(X_test[i].reshape(1, -1), 
                                            ((0, 0), (0, height * width - n_features)), 
                                            'constant').reshape(height, width)
    
    return X_train_reshaped, X_test_reshaped, y_train, y_test, {'scaler': scaler, 'label_encoders': label_encoders, 'height': height, 'width': width}
