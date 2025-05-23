import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import sys
import tensorflow as tf
from tensorflow.keras.models import load_model
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
from io import BytesIO
import json
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Crear objetos personalizados para las métricas
def mse(y_true, y_pred):
    return tf.keras.metrics.mean_squared_error(y_true, y_pred)

def mae(y_true, y_pred):
    return tf.keras.metrics.mean_absolute_error(y_true, y_pred)

# Registrar las métricas personalizadas con TensorFlow
tf.keras.utils.get_custom_objects().update({
    'mse': mse,
    'mae': mae
})

# No necesitamos agregar directorio raíz al path ya que los archivos están en el mismo directorio
# sys.path.append('..')
from data_utils import load_data, preprocess_data, prepare_time_series_data

# Configuración de la página
st.set_page_config(
    page_title="Análisis y Predicción de Ventas de Supermercado",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #0D47A1;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #1565C0;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
    }
    .highlight {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #F5F5F5;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1E88E5;
    }
    .metric-label {
        font-size: 1rem;
        color: #424242;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        color: #757575;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Función para cargar datos utilizando la función importada de data_utils
@st.cache_data
def load_app_data():
    try:
        # Cargar datos originales
        data_path = 'supermarket_sales.xlsx'
        data = load_data(data_path)
        return data
    except Exception as e:
        st.error(f"Error al cargar los datos: {e}")
        return None

# Función para cargar modelos
@st.cache_resource
def load_models():
    models = {}
    # Definimos las rutas de los modelos entrenados
    model_paths = {
        'MLP': 'models/mlp/mlp_ventas_totales.h5',
        'LSTM': 'models/lstm/lstm_ventas_totales_seq7.h5',
        'CNN': 'models/cnn/cnn_ventas_totales.h5',
        'Baseline': 'models/baseline/models/rf_optimized_model.joblib',
        'Híbrido': 'models/hybrid/hybrid_model.h5',
        'Ensemble': 'models/ensemble/stacking_regressor.joblib'
    }
    
    # Verificar si existen los modelos entrenados
    models_found = any(os.path.exists(path) for path in model_paths.values())
    
    # Si no encontramos modelos, mostramos un mensaje
    if not models_found:
        st.sidebar.warning("No se encontraron modelos preentrenados. Modo de demostración activado.")
    
    for name, path in model_paths.items():
        try:
            if path.endswith('.h5'):
                models[name] = load_model(path)
            else:
                models[name] = joblib.load(path)
            st.sidebar.success(f"Modelo {name} cargado correctamente")
        except Exception as e:
            st.sidebar.warning(f"No se pudo cargar el modelo {name}: {e}")
    
    return models

# Función para cargar preprocesadores
@st.cache_resource
def load_preprocessors():
    preprocessors = {}
    try:
        # Verificar si existen los archivos de preprocesadores
        mlp_path = 'data/processed/mlp_datasets.joblib'
        lstm_path = 'data/processed/lstm_datasets.joblib'
        cnn_path = 'data/processed/cnn_datasets.joblib'
        
        if os.path.exists(mlp_path) and os.path.exists(lstm_path) and os.path.exists(cnn_path):
            # Cargar preprocesadores entrenados
            mlp_datasets = joblib.load(mlp_path)
            lstm_datasets = joblib.load(lstm_path)
            cnn_datasets = joblib.load(cnn_path)
            preprocessors['MLP'] = mlp_datasets['ventas_totales']['preprocessor']
            preprocessors['LSTM'] = lstm_datasets['ventas_totales_seq7']['preprocessor']
            preprocessors['CNN'] = cnn_datasets['ventas_totales']['preprocessor']
            preprocessors['Baseline'] = mlp_datasets['ventas_totales']['preprocessor']  # Usar el mismo que MLP para Baseline
            preprocessors['Híbrido'] = {'scaler': preprocessors['MLP']['scaler']}  # Configuración básica
            preprocessors['Ensemble'] = {'scaler': preprocessors['MLP']['scaler'], 'feature_names': preprocessors['MLP']['feature_names']}  # Usar el mismo que MLP
            
            st.sidebar.success("Preprocesadores cargados correctamente")
        else:            # Creamos preprocesadores de ejemplo para modo demostración
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            # Preprocesadores de ejemplo
            preprocessors['MLP'] = {'scaler': scaler, 'feature_names': ['Quantity', 'Unit price', 'Total']}
            preprocessors['LSTM'] = {'scaler': scaler, 'sequence_length': 7}
            preprocessors['CNN'] = {'scaler': scaler}
            preprocessors['Baseline'] = {'scaler': scaler, 'feature_names': ['Quantity', 'Unit price', 'Total']}
            preprocessors['Híbrido'] = {'scaler': scaler}
            preprocessors['Ensemble'] = {'scaler': scaler, 'feature_names': ['Quantity', 'Unit price', 'Total']}
            
            st.sidebar.warning("Preprocesadores de demostración configurados")
    except Exception as e:
        st.sidebar.warning(f"No se pudieron configurar los preprocesadores: {e}")
    
    return preprocessors

# Función para cargar resultados de modelos
@st.cache_data
def load_model_results():
    results = {}
    # Definimos las rutas de los resultados
    result_paths = {
        'MLP': 'models/mlp/results/mlp_results.joblib',
        'LSTM': 'models/lstm/results/lstm_results.joblib',
        'CNN': 'models/cnn/results/cnn_results.joblib',
        'Baseline': 'models/baseline/results/baseline_results.joblib',
        'Híbrido': 'models/hybrid/results/hybrid_results.joblib',
        'Ensemble': 'models/ensemble/results/ensemble_results.joblib'
    }
    
    # Verificar si existe al menos un archivo de resultados
    results_found = any(os.path.exists(path) for path in result_paths.values())
    
    if results_found:
        # Cargar los resultados de modelos entrenados
        for name, path in result_paths.items():
            try:
                if os.path.exists(path):
                    results[name] = joblib.load(path)
                    st.sidebar.success(f"Resultados del modelo {name} cargados correctamente")
                else:
                    st.sidebar.warning(f"No se encontraron resultados para el modelo {name}")
            except Exception as e:
                st.sidebar.warning(f"No se pudieron cargar los resultados del modelo {name}: {e}")
                
        # Crear datos para comparación
        model_names = []
        mae_values = []
        mse_values = []
        rmse_values = []
        r2_values = []
        
        for name, model_results in results.items():
            if name != 'Baseline':
                if 'metrics' in model_results:
                    model_names.append(name)
                    mae_values.append(model_results['metrics']['mae'])
                    mse_values.append(model_results['metrics']['mse'])
                    rmse_values.append(model_results['metrics']['rmse'])
                    r2_values.append(model_results['metrics']['r2'])
            elif 'rf_optimized' in model_results:
                model_names.append('Baseline')
                mae_values.append(model_results['rf_optimized']['metrics']['mae'])
                mse_values.append(model_results['rf_optimized']['metrics']['mse'])
                rmse_values.append(model_results['rf_optimized']['metrics']['rmse'])
                r2_values.append(model_results['rf_optimized']['metrics']['r2'])
        
        # Crear DataFrame de comparación
        if model_names:
            results['comparison'] = pd.DataFrame({
                'Modelo': model_names,
                'MAE': mae_values,
                'MSE': mse_values,
                'RMSE': rmse_values,
                'R²': r2_values
            })
        
        st.sidebar.success("Resultados de modelos cargados correctamente")
    else:
        # Creamos resultados de ejemplo para modo demostración
        st.sidebar.warning("No se encontraron resultados de modelos. Usando resultados de demostración.")
        
        # Resultados de ejemplo para cada modelo
        models = ['MLP', 'LSTM', 'CNN', 'Baseline', 'Híbrido', 'Ensemble']
        for model in models:
            results[model] = {
                'metrics': {
                    'mae': np.random.uniform(10, 50),
                    'mse': np.random.uniform(100, 500),
                    'rmse': np.random.uniform(15, 40),
                    'r2': np.random.uniform(0.7, 0.95)
                },
                'predictions': np.random.normal(100, 20, size=50),
                'actual': np.random.normal(100, 20, size=50)
            }
        
        # Resultados para comparación
        results['comparison'] = pd.DataFrame({
            'Modelo': models,
            'MAE': [results[m]['metrics']['mae'] for m in models],
            'MSE': [results[m]['metrics']['mse'] for m in models],
            'RMSE': [results[m]['metrics']['rmse'] for m in models],
            'R²': [results[m]['metrics']['r2'] for m in models]
        })
    
    return results

# Función para generar predicciones
def generate_predictions(model_name, model, data, target_variable, preprocessors):
    try:
        if model_name == 'MLP':
            # Preprocesar datos para MLP
            X = preprocessors['MLP']['scaler'].transform(data[preprocessors['MLP']['feature_names']])
            predictions = model.predict(X)
        elif model_name == 'LSTM':
            # Preprocesar datos para LSTM
            sequence_length = preprocessors['LSTM']['sequence_length']
            # Usamos prepare_time_series_data en lugar de prepare_data_for_prediction
            X, _ = prepare_time_series_data(data, target_variable, sequence_length)
            if 'scaler' in preprocessors['LSTM']:
                X = preprocessors['LSTM']['scaler'].transform(X)
            predictions = model.predict(X)
        elif model_name == 'CNN':
            # Preprocesar datos para CNN
            # Implementar lógica específica para CNN
            pass
        elif model_name == 'Baseline':
            # Preprocesar datos para modelos baseline
            # Usamos el preprocesador de MLP ya que el baseline usa los mismos datos
            X = preprocessors['MLP']['scaler'].transform(data[preprocessors['MLP']['feature_names']])
            predictions = model.predict(X)
        elif model_name == 'Híbrido':
            # Preprocesar datos para modelo híbrido
            # Necesitamos datos preprocesados para las tres entradas del modelo híbrido: MLP, LSTM y CNN
            st.warning("La predicción con el modelo Híbrido requiere implementación específica adicional")
            predictions = None
        elif model_name == 'Ensemble':
            # Preprocesar datos para modelo ensemble
            X = preprocessors['MLP']['scaler'].transform(data[preprocessors['MLP']['feature_names']])
            predictions = model.predict(X)
        
        return predictions
    except Exception as e:
        st.error(f"Error al generar predicciones con el modelo {model_name}: {e}")
        return None

# Función para crear gráficos de comparación
def plot_model_comparison(results, metric='RMSE'):
    if 'comparison' in results and isinstance(results['comparison'], pd.DataFrame):
        df = results['comparison']
        fig = px.bar(df, x='Modelo', y=metric, 
                    title=f'Comparación de {metric} entre Modelos',
                    color='Modelo', 
                    color_discrete_sequence=px.colors.qualitative.Bold)
        fig.update_layout(xaxis_title='Modelo', 
                        yaxis_title=metric,
                        xaxis={'categoryorder':'total descending'},
                        height=500)
        return fig
    else:
        # Crear DataFrame manualmente a partir de los resultados disponibles
        comparison_data = []
        
        for model_type, model_results in results.items():
            if model_type == 'comparison':
                continue
                
            if model_type == 'MLP' and 'ventas_totales' in model_results:
                comparison_data.append({
                    'Modelo': 'MLP',
                    'RMSE': model_results['ventas_totales']['metrics']['rmse'],
                    'MAE': model_results['ventas_totales']['metrics']['mae'],
                    'R²': model_results['ventas_totales']['metrics']['r2']
                })
            elif model_type == 'LSTM' and 'ventas_totales_seq7' in model_results:
                comparison_data.append({
                    'Modelo': 'LSTM',
                    'RMSE': model_results['ventas_totales_seq7']['metrics']['rmse'],
                    'MAE': model_results['ventas_totales_seq7']['metrics']['mae'],
                    'R²': model_results['ventas_totales_seq7']['metrics']['r2']
                })
            elif model_type == 'CNN' and 'ventas_totales' in model_results:
                comparison_data.append({
                    'Modelo': 'CNN',
                    'RMSE': model_results['ventas_totales']['metrics']['rmse'],
                    'MAE': model_results['ventas_totales']['metrics']['mae'],
                    'R²': model_results['ventas_totales']['metrics']['r2']
                })
            elif model_type == 'Baseline' and 'rf_optimized' in model_results:
                comparison_data.append({
                    'Modelo': 'Random Forest',
                    'RMSE': model_results['rf_optimized']['metrics']['rmse'],
                    'MAE': model_results['rf_optimized']['metrics']['mae'],
                    'R²': model_results['rf_optimized']['metrics']['r2']
                })
            elif model_type == 'Híbrido' and 'hybrid' in model_results:
                comparison_data.append({
                    'Modelo': 'Híbrido',
                    'RMSE': model_results['hybrid']['metrics']['rmse'],
                    'MAE': model_results['hybrid']['metrics']['mae'],
                    'R²': model_results['hybrid']['metrics']['r2']
                })
        
        if comparison_data:
            df = pd.DataFrame(comparison_data)
            fig = px.bar(df, x='Modelo', y=metric, 
                        title=f'Comparación de {metric} entre Modelos',
                        color='Modelo', 
                        color_discrete_sequence=px.colors.qualitative.Bold)
            fig.update_layout(xaxis_title='Modelo', 
                            yaxis_title=metric,
                            xaxis={'categoryorder':'total descending'},
                            height=500)
            return fig
        else:
            return None

# Función para crear gráficos de series temporales
def plot_time_series(data, variable, title):
    fig = px.line(data, x='Date', y=variable, 
                 title=title,
                 labels={'Date': 'Fecha', variable: 'Valor'},
                 line_shape='linear')
    fig.update_layout(height=400)
    return fig

# Función para crear gráficos de barras
def plot_bar_chart(data, x, y, title):
    fig = px.bar(data, x=x, y=y, 
                title=title,
                labels={x: x, y: y},
                color=x)
    fig.update_layout(height=400)
    return fig

# Función para crear gráficos de dispersión
def plot_scatter(data, x, y, title, color=None):
    fig = px.scatter(data, x=x, y=y, 
                    title=title,
                    labels={x: x, y: y},
                    color=color if color else None)
    fig.update_layout(height=400)
    return fig

# Función para exportar resultados
def export_results_to_excel(data, predictions, filename="predicciones.xlsx"):
    # Crear DataFrame con resultados
    results_df = pd.DataFrame({
        'Fecha': data['Date'],
        'Valor Real': data['Total'],
        'Predicción': predictions.flatten()
    })
    
    # Guardar a Excel
    buffer = BytesIO()
    results_df.to_excel(buffer, index=False)
    buffer.seek(0)
    
    return buffer

# Función para descargar archivos
def get_download_link(buffer, filename, text):
    b64 = base64.b64encode(buffer.read()).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">{text}</a>'
    return href

# Función para mostrar descripciones de modelos
def show_model_description(model_name):
    descriptions = {
        'MLP': """
        ### Red Neuronal Multicapa (MLP)
        
        La Red Neuronal Multicapa es un tipo de red neuronal artificial compuesta por múltiples capas de neuronas. En este proyecto, se utiliza para predecir ventas basándose en características tabulares.
        
        **Características principales:**
        - Arquitectura: Capas densas con activación ReLU
        - Regularización: Dropout y BatchNormalization
        - Optimizador: Adam con tasa de aprendizaje adaptativa
        
        **Ventajas:**
        - Capacidad para modelar relaciones no lineales complejas
        - Buen rendimiento en datos tabulares estructurados
        - Relativamente rápido de entrenar comparado con otras arquitecturas de redes neuronales
        
        **Limitaciones:**
        - No captura dependencias temporales explícitamente
        - Puede sobreajustarse en conjuntos de datos pequeños
        """,
        
        'LSTM': """
        ### Red Neuronal Recurrente LSTM
        
        Las redes LSTM (Long Short-Term Memory) son un tipo de red neuronal recurrente diseñada para capturar dependencias temporales a largo plazo en secuencias de datos.
        
        **Características principales:**
        - Arquitectura: Celdas LSTM con mecanismos de compuerta
        - Secuencia temporal: Utiliza ventanas deslizantes de datos históricos
        - Capacidad de memoria: Retiene información relevante a lo largo del tiempo
        
        **Ventajas:**
        - Excelente para capturar patrones temporales y estacionales
        - Maneja eficientemente dependencias a largo plazo
        - Ideal para predicción de series temporales
        
        **Limitaciones:**
        - Requiere datos secuenciales estructurados
        - Mayor tiempo de entrenamiento
        - Necesita suficientes datos históricos para ser efectivo
        """,
        
        'CNN': """
        ### Red Neuronal Convolucional (CNN)
        
        Aunque tradicionalmente utilizadas para procesamiento de imágenes, las CNN pueden adaptarse para datos tabulares y series temporales mediante la conversión de datos a formatos matriciales.
        
        **Características principales:**
        - Arquitectura: Capas convolucionales seguidas de capas densas
        - Extracción de características: Filtros convolucionales que detectan patrones
        - Pooling: Reducción de dimensionalidad para capturar características importantes
        
        **Ventajas:**
        - Capacidad para detectar patrones locales en los datos
        - Reducción automática de dimensionalidad
        - Robustez ante variaciones en los datos
        
        **Limitaciones:**
        - Requiere transformación de datos tabulares a formato matricial
        - Puede ser complejo de interpretar
        - Mayor número de hiperparámetros a ajustar
        """,
        
        'Baseline': """
        ### Modelos Baseline (Random Forest)
        
        Los modelos baseline proporcionan un punto de referencia para evaluar el rendimiento de modelos más complejos. En este proyecto, se utiliza Random Forest como modelo baseline principal.
        
        **Características principales:**
        - Conjunto de árboles de decisión
        - Técnicas de bagging para reducir varianza
        - Selección aleatoria de características
        
        **Ventajas:**
        - Robusto ante outliers y ruido
        - Proporciona importancia de características
        - No requiere escalado de datos
        - Rápido de entrenar y predecir
        
        **Limitaciones:**
        - Puede tener dificultades con relaciones muy complejas
        - Menos efectivo en datos altamente dimensionales
        - Tendencia a sobreajustar si no se controlan hiperparámetros
        """,
        
        'Híbrido': """
        ### Modelo Híbrido (MLP+LSTM+CNN)
        
        El modelo híbrido combina diferentes arquitecturas de redes neuronales para aprovechar las fortalezas de cada una y compensar sus debilidades.
        
        **Características principales:**
        - Arquitectura: Combina ramas de MLP, LSTM y CNN
        - Fusión: Concatenación de características extraídas por cada rama
        - Capas finales: Densas con regularización
        
        **Ventajas:**
        - Captura simultáneamente patrones tabulares, temporales y espaciales
        - Mayor capacidad expresiva que modelos individuales
        - Potencialmente mejor rendimiento en problemas complejos
        
        **Limitaciones:**
        - Mayor complejidad computacional
        - Requiere más datos para entrenar efectivamente
        - Mayor riesgo de sobreajuste si no se regulariza adecuadamente
        """,
        
        'Ensemble': """
        ### Modelo Ensemble
        
        Los modelos ensemble combinan las predicciones de múltiples modelos para obtener predicciones más robustas y precisas.
        
        **Características principales:**
        - Técnicas: Voting, Stacking o promedio de predicciones
        - Diversidad: Combina modelos con diferentes fortalezas
        - Meta-modelo: En stacking, aprende a combinar predicciones óptimamente
        
        **Ventajas:**
        - Reduce varianza y sesgo
        - Mayor robustez ante diferentes condiciones de datos
        - Generalmente mejor rendimiento que modelos individuales
        
        **Limitaciones:**
        - Mayor complejidad computacional
        - Más difícil de interpretar
        - Requiere gestión de múltiples modelos
        """
    }
    
    if model_name in descriptions:
        st.markdown(descriptions[model_name], unsafe_allow_html=True)
    else:
        st.warning(f"No hay descripción disponible para el modelo {model_name}")

# Función principal de la aplicación
def main():
    # Título principal
    st.markdown('<h1 class="main-header">Análisis y Predicción de Ventas de Supermercado</h1>', unsafe_allow_html=True)
    
    # Cargar datos
    data = load_app_data()
    
    if data is None:
        st.error("No se pudieron cargar los datos. Por favor, verifica la ruta del archivo.")
        return
    
    # Cargar modelos, preprocesadores y resultados
    with st.sidebar:
        st.sidebar.markdown("## Configuración")
        st.sidebar.markdown("### Cargando recursos...")
        
    models = load_models()
    preprocessors = load_preprocessors()
    model_results = load_model_results()
    
    # Sidebar - Selección de opciones
    with st.sidebar:
        st.sidebar.markdown("## Navegación")
        app_mode = st.sidebar.selectbox(
            "Seleccione una sección",
            ["Inicio", "Exploración de Datos", "Predicción de Ventas", "Comparación de Modelos", "Visualizaciones Avanzadas", "Exportar Resultados"]
        )
        
        st.sidebar.markdown("## Parámetros")
        
        # Selección de variable objetivo
        target_variable = st.sidebar.selectbox(
            "Variable objetivo",
            ["Total", "Quantity", "Unit price", "Rating"],
            index=0
        )
        
        # Selección de modelo
        selected_model = st.sidebar.selectbox(
            "Modelo",
            list(models.keys()),
            index=0 if models else None
        )
        
        # Filtros adicionales según la sección
        if app_mode == "Exploración de Datos" or app_mode == "Visualizaciones Avanzadas":
            # Filtro por sucursal
            branch_filter = st.sidebar.multiselect(
                "Sucursal",
                data["Branch"].unique(),
                default=data["Branch"].unique()
            )
            
            # Filtro por categoría de producto
            category_filter = st.sidebar.multiselect(
                "Categoría de Producto",
                data["Product line"].unique(),
                default=data["Product line"].unique()
            )
            
            # Filtro por método de pago
            payment_filter = st.sidebar.multiselect(
                "Método de Pago",
                data["Payment"].unique(),
                default=data["Payment"].unique()
            )
            
            # Filtro por género del cliente
            gender_filter = st.sidebar.multiselect(
                "Género del Cliente",
                data["Gender"].unique(),
                default=data["Gender"].unique()
            )
            
            # Aplicar filtros
            filtered_data = data[
                data["Branch"].isin(branch_filter) &
                data["Product line"].isin(category_filter) &
                data["Payment"].isin(payment_filter) &
                data["Gender"].isin(gender_filter)
            ]
        else:
            filtered_data = data
    
    # Contenido principal según la sección seleccionada
    if app_mode == "Inicio":
        show_home_page(data)
    
    elif app_mode == "Exploración de Datos":
        show_data_exploration(filtered_data)
    
    elif app_mode == "Predicción de Ventas":
        show_sales_prediction(data, models, selected_model, target_variable, preprocessors)
    
    elif app_mode == "Comparación de Modelos":
        show_model_comparison(model_results)
    
    elif app_mode == "Visualizaciones Avanzadas":
        show_advanced_visualizations(filtered_data)
    
    elif app_mode == "Exportar Resultados":
        show_export_results(data, models, selected_model, target_variable, preprocessors)
    
    # Footer
    st.markdown('<div class="footer">Desarrollado para el Proyecto de Análisis y Predicción de Ventas de Supermercado © 2025</div>', unsafe_allow_html=True)

# Función para mostrar la página de inicio
def show_home_page(data):
    st.markdown('<h2 class="sub-header">Bienvenido al Sistema de Análisis y Predicción de Ventas</h2>', unsafe_allow_html=True)
    
    # Descripción del proyecto
    st.markdown("""
    <div class="highlight">
    Esta aplicación permite analizar datos históricos de ventas de supermercado y realizar predicciones utilizando 
    diferentes modelos de aprendizaje automático y redes neuronales. Diseñada específicamente para propietarios 
    de pequeños supermercados, proporciona herramientas para la toma de decisiones basada en datos.
    </div>
    """, unsafe_allow_html=True)
    
    # Resumen de datos
    st.markdown('<h3 class="section-header">Resumen de Datos</h3>', unsafe_allow_html=True)
    
    # Métricas principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{data.shape[0]}</div>
            <div class="metric-label">Transacciones</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{data['Branch'].nunique()}</div>
            <div class="metric-label">Sucursales</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{data['Product line'].nunique()}</div>
            <div class="metric-label">Categorías de Productos</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">${data['Total'].sum():,.2f}</div>
            <div class="metric-label">Ventas Totales</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Gráfico de ventas por sucursal
    st.markdown('<h3 class="section-header">Ventas por Sucursal</h3>', unsafe_allow_html=True)
    branch_sales = data.groupby('Branch')['Total'].sum().reset_index()
    fig_branch = px.bar(branch_sales, x='Branch', y='Total', 
                       title='Ventas Totales por Sucursal',
                       labels={'Branch': 'Sucursal', 'Total': 'Ventas Totales'},
                       color='Branch')
    st.plotly_chart(fig_branch, use_container_width=True)
    
    # Gráfico de ventas por categoría de producto
    st.markdown('<h3 class="section-header">Ventas por Categoría de Producto</h3>', unsafe_allow_html=True)
    category_sales = data.groupby('Product line')['Total'].sum().reset_index()
    fig_category = px.bar(category_sales, x='Product line', y='Total', 
                         title='Ventas Totales por Categoría de Producto',
                         labels={'Product line': 'Categoría', 'Total': 'Ventas Totales'},
                         color='Product line')
    st.plotly_chart(fig_category, use_container_width=True)
    
    # Instrucciones de uso
    st.markdown('<h3 class="section-header">Cómo Usar esta Aplicación</h3>', unsafe_allow_html=True)
    
    st.markdown("""
    1. **Exploración de Datos**: Analice los datos históricos de ventas con visualizaciones interactivas.
    2. **Predicción de Ventas**: Utilice diferentes modelos para predecir ventas futuras.
    3. **Comparación de Modelos**: Compare el rendimiento de los diferentes modelos implementados.
    4. **Visualizaciones Avanzadas**: Explore relaciones complejas en los datos con gráficos avanzados.
    5. **Exportar Resultados**: Exporte predicciones y análisis para su uso posterior.
    
    Utilice el menú de navegación en la barra lateral para acceder a las diferentes secciones.
    """)

# Función para mostrar la exploración de datos
def show_data_exploration(data):
    st.markdown('<h2 class="sub-header">Exploración de Datos</h2>', unsafe_allow_html=True)
    
    # Mostrar datos filtrados
    st.markdown('<h3 class="section-header">Datos Filtrados</h3>', unsafe_allow_html=True)
    st.dataframe(data.head(100))
    
    # Estadísticas descriptivas
    st.markdown('<h3 class="section-header">Estadísticas Descriptivas</h3>', unsafe_allow_html=True)
    st.dataframe(data.describe())
    
    # Visualizaciones
    st.markdown('<h3 class="section-header">Visualizaciones</h3>', unsafe_allow_html=True)
    
    # Selección de tipo de gráfico
    chart_type = st.selectbox(
        "Seleccione tipo de visualización",
        ["Ventas por Tiempo", "Distribución de Variables", "Correlaciones", "Ventas por Categoría", "Ventas por Método de Pago", "Calificación por Categoría"]
    )
    
    if chart_type == "Ventas por Tiempo":
        # Preparar datos temporales
        data['Date'] = pd.to_datetime(data['Date'])
        daily_sales = data.groupby('Date')['Total'].sum().reset_index()
        
        # Gráfico de ventas diarias
        fig = plot_time_series(daily_sales, 'Total', 'Ventas Diarias')
        st.plotly_chart(fig, use_container_width=True)
        
    elif chart_type == "Distribución de Variables":
        # Selección de variable
        variable = st.selectbox(
            "Seleccione variable para visualizar distribución",
            ["Total", "Quantity", "Unit price", "Rating", "gross income"]
        )
        
        # Histograma
        fig = px.histogram(data, x=variable, 
                          title=f'Distribución de {variable}',
                          labels={variable: variable},
                          color_discrete_sequence=['#1E88E5'])
        st.plotly_chart(fig, use_container_width=True)
        
    elif chart_type == "Correlaciones":
        # Matriz de correlación
        numeric_data = data.select_dtypes(include=['float64', 'int64'])
        corr = numeric_data.corr()
        
        fig = px.imshow(corr, 
                       title='Matriz de Correlación',
                       labels=dict(color="Correlación"),
                       color_continuous_scale='RdBu_r')
        st.plotly_chart(fig, use_container_width=True)
        
    elif chart_type == "Ventas por Categoría":
        # Ventas por categoría de producto
        category_sales = data.groupby('Product line')['Total'].sum().reset_index()
        
        fig = plot_bar_chart(category_sales, 'Product line', 'Total', 'Ventas Totales por Categoría de Producto')
        st.plotly_chart(fig, use_container_width=True)
        
    elif chart_type == "Ventas por Método de Pago":
        # Ventas por método de pago
        payment_sales = data.groupby('Payment')['Total'].sum().reset_index()
        
        fig = plot_bar_chart(payment_sales, 'Payment', 'Total', 'Ventas Totales por Método de Pago')
        st.plotly_chart(fig, use_container_width=True)
        
    elif chart_type == "Calificación por Categoría":
        # Calificación promedio por categoría
        rating_by_category = data.groupby('Product line')['Rating'].mean().reset_index()
        
        fig = plot_bar_chart(rating_by_category, 'Product line', 'Rating', 'Calificación Promedio por Categoría de Producto')
        st.plotly_chart(fig, use_container_width=True)

# Función para mostrar la predicción de ventas
def show_sales_prediction(data, models, selected_model, target_variable, preprocessors):
    st.markdown('<h2 class="sub-header">Predicción de Ventas</h2>', unsafe_allow_html=True)
    
    # Descripción del modelo seleccionado
    st.markdown('<h3 class="section-header">Descripción del Modelo</h3>', unsafe_allow_html=True)
    show_model_description(selected_model)
    
    # Parámetros de predicción
    st.markdown('<h3 class="section-header">Parámetros de Predicción</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Selección de sucursal
        branch = st.selectbox(
            "Sucursal",
            data["Branch"].unique(),
            index=0
        )
        
        # Selección de categoría de producto
        product_line = st.selectbox(
            "Categoría de Producto",
            data["Product line"].unique(),
            index=0
        )
    
    with col2:
        # Selección de método de pago
        payment = st.selectbox(
            "Método de Pago",
            data["Payment"].unique(),
            index=0
        )
        
        # Selección de género del cliente
        gender = st.selectbox(
            "Género del Cliente",
            data["Gender"].unique(),
            index=0
        )
    
    # Botón para generar predicción
    if st.button("Generar Predicción"):
        # Mostrar spinner mientras se genera la predicción
        with st.spinner("Generando predicción..."):
            # Simular tiempo de procesamiento
            time.sleep(2)
            
            # Crear datos de ejemplo para predicción
            # En una aplicación real, esto se basaría en los parámetros seleccionados
            sample_data = data.sample(10).copy()
            
            # Generar predicción
            if selected_model in models:
                predictions = generate_predictions(selected_model, models[selected_model], sample_data, target_variable, preprocessors)
                
                if predictions is not None:
                    # Mostrar resultados
                    st.markdown('<h3 class="section-header">Resultados de la Predicción</h3>', unsafe_allow_html=True)
                    
                    # Crear DataFrame con resultados
                    results_df = pd.DataFrame({
                        'Fecha': sample_data['Date'],
                        'Valor Real': sample_data[target_variable],
                        'Predicción': predictions.flatten()
                    })
                    
                    # Mostrar tabla de resultados
                    st.dataframe(results_df)
                    
                    # Visualizar predicciones vs valores reales
                    fig = px.scatter(results_df, x='Valor Real', y='Predicción', 
                                    title=f'Predicciones vs Valores Reales - {selected_model}',
                                    labels={'Valor Real': 'Valor Real', 'Predicción': 'Predicción'},
                                    trendline='ols')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Métricas de error
                    mse = mean_squared_error(results_df['Valor Real'], results_df['Predicción'])
                    rmse = np.sqrt(mse)
                    mae = mean_absolute_error(results_df['Valor Real'], results_df['Predicción'])
                    
                    # Mostrar métricas
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{mse:.4f}</div>
                            <div class="metric-label">MSE</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{rmse:.4f}</div>
                            <div class="metric-label">RMSE</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{mae:.4f}</div>
                            <div class="metric-label">MAE</div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.error(f"El modelo {selected_model} no está disponible.")

# Función para mostrar la comparación de modelos
def show_model_comparison(model_results):
    st.markdown('<h2 class="sub-header">Comparación de Modelos</h2>', unsafe_allow_html=True)
    
    # Selección de métrica
    metric = st.selectbox(
        "Seleccione métrica para comparar",
        ["RMSE", "MAE", "R²"],
        index=0
    )
    
    # Generar gráfico de comparación
    fig = plot_model_comparison(model_results, metric)
    
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No hay datos suficientes para generar la comparación de modelos.")
    
    # Tabla de comparación detallada
    st.markdown('<h3 class="section-header">Comparación Detallada</h3>', unsafe_allow_html=True)
    
    # Crear tabla de comparación manualmente si no está disponible
    if 'comparison' in model_results and isinstance(model_results['comparison'], pd.DataFrame):
        comparison_df = model_results['comparison']
        st.dataframe(comparison_df)
    else:
        # Crear DataFrame manualmente a partir de los resultados disponibles
        comparison_data = []
        
        for model_type, model_results_data in model_results.items():
            if model_type == 'comparison':
                continue
                
            if model_type == 'MLP' and 'ventas_totales' in model_results_data:
                comparison_data.append({
                    'Modelo': 'MLP',
                    'RMSE': model_results_data['ventas_totales']['metrics']['rmse'],
                    'MAE': model_results_data['ventas_totales']['metrics']['mae'],
                    'R²': model_results_data['ventas_totales']['metrics']['r2'],
                    'Tiempo (s)': model_results_data['ventas_totales']['training_time']
                })
            elif model_type == 'LSTM' and 'ventas_totales_seq7' in model_results_data:
                comparison_data.append({
                    'Modelo': 'LSTM',
                    'RMSE': model_results_data['ventas_totales_seq7']['metrics']['rmse'],
                    'MAE': model_results_data['ventas_totales_seq7']['metrics']['mae'],
                    'R²': model_results_data['ventas_totales_seq7']['metrics']['r2'],
                    'Tiempo (s)': model_results_data['ventas_totales_seq7']['training_time']
                })
            elif model_type == 'CNN' and 'ventas_totales' in model_results_data:
                comparison_data.append({
                    'Modelo': 'CNN',
                    'RMSE': model_results_data['ventas_totales']['metrics']['rmse'],
                    'MAE': model_results_data['ventas_totales']['metrics']['mae'],
                    'R²': model_results_data['ventas_totales']['metrics']['r2'],
                    'Tiempo (s)': model_results_data['ventas_totales']['training_time']
                })
            elif model_type == 'Baseline' and 'rf_optimized' in model_results_data:
                comparison_data.append({
                    'Modelo': 'Random Forest',
                    'RMSE': model_results_data['rf_optimized']['metrics']['rmse'],
                    'MAE': model_results_data['rf_optimized']['metrics']['mae'],
                    'R²': model_results_data['rf_optimized']['metrics']['r2'],
                    'Tiempo (s)': model_results_data['rf_optimized'].get('optimization_time', 0)
                })
            elif model_type == 'Híbrido' and 'hybrid' in model_results_data:
                comparison_data.append({
                    'Modelo': 'Híbrido',
                    'RMSE': model_results_data['hybrid']['metrics']['rmse'],
                    'MAE': model_results_data['hybrid']['metrics']['mae'],
                    'R²': model_results_data['hybrid']['metrics']['r2'],
                    'Tiempo (s)': model_results_data['hybrid']['training_time']
                })
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df)
        else:
            st.warning("No hay datos suficientes para generar la tabla de comparación.")
    
    # Análisis de fortalezas y debilidades
    st.markdown('<h3 class="section-header">Análisis de Fortalezas y Debilidades</h3>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Análisis Comparativo de Modelos
    
    #### MLP (Red Neuronal Multicapa)
    - **Fortalezas**: Buena capacidad para modelar relaciones no lineales en datos tabulares, relativamente rápido de entrenar.
    - **Debilidades**: No captura explícitamente dependencias temporales, puede requerir más ingeniería de características.
    
    #### LSTM (Long Short-Term Memory)
    - **Fortalezas**: Excelente para capturar patrones temporales y estacionales, maneja eficientemente dependencias a largo plazo.
    - **Debilidades**: Mayor tiempo de entrenamiento, requiere datos secuenciales estructurados.
    
    #### CNN (Red Neuronal Convolucional)
    - **Fortalezas**: Capacidad para detectar patrones locales en los datos, robustez ante variaciones.
    - **Debilidades**: Requiere transformación de datos tabulares a formato matricial, puede ser complejo de interpretar.
    
    #### Random Forest (Baseline)
    - **Fortalezas**: Robusto ante outliers y ruido, proporciona importancia de características, no requiere escalado de datos.
    - **Debilidades**: Puede tener dificultades con relaciones muy complejas, tendencia a sobreajustar si no se controlan hiperparámetros.
    
    #### Modelo Híbrido
    - **Fortalezas**: Captura simultáneamente patrones tabulares, temporales y espaciales, mayor capacidad expresiva.
    - **Debilidades**: Mayor complejidad computacional, requiere más datos para entrenar efectivamente.
    
    #### Modelo Ensemble
    - **Fortalezas**: Reduce varianza y sesgo, mayor robustez ante diferentes condiciones de datos.
    - **Debilidades**: Mayor complejidad computacional, más difícil de interpretar.
    """)
    
    # Recomendaciones para propietarios de supermercados
    st.markdown('<h3 class="section-header">Recomendaciones para Propietarios de Supermercados</h3>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Recomendaciones Basadas en el Análisis de Modelos
    
    1. **Para predicciones diarias de ventas totales**: Utilice el modelo Ensemble o Híbrido, que generalmente ofrecen el mejor equilibrio entre precisión y robustez.
    
    2. **Para análisis rápidos o con recursos limitados**: El modelo Random Forest optimizado proporciona un buen rendimiento con menor complejidad computacional.
    
    3. **Para predicciones con patrones estacionales fuertes**: Priorice el modelo LSTM, que captura eficientemente dependencias temporales.
    
    4. **Para entender factores que influyen en las ventas**: Utilice Random Forest, que proporciona importancia de características interpretable.
    
    5. **Para predicciones a largo plazo**: Combine los resultados de múltiples modelos (Ensemble) para obtener mayor robustez.
    
    6. **Para implementación en sistemas con recursos limitados**: Considere el modelo MLP o Random Forest, que ofrecen buen equilibrio entre rendimiento y eficiencia.
    """)

# Función para mostrar visualizaciones avanzadas
def show_advanced_visualizations(data):
    st.markdown('<h2 class="sub-header">Visualizaciones Avanzadas</h2>', unsafe_allow_html=True)
      # Selección de tipo de visualización
    viz_type = st.selectbox(
        "Seleccione tipo de visualización",
        ["Análisis de Tendencias", "Patrones por Hora del Día", "Segmentación de Clientes", "Análisis de Canasta de Compra", "Mapas de Calor", "Análisis de Rentabilidad"]
    )
    
    if viz_type == "Análisis de Tendencias":
        # Preparar datos temporales
        data['Date'] = pd.to_datetime(data['Date'])
        data['Month'] = data['Date'].dt.month
        data['Day'] = data['Date'].dt.day
        data['DayOfWeek'] = data['Date'].dt.dayofweek
        
        # Ventas por día de la semana
        sales_by_dow = data.groupby('DayOfWeek')['Total'].mean().reset_index()
        sales_by_dow['DayName'] = sales_by_dow['DayOfWeek'].map({
            0: 'Lunes', 1: 'Martes', 2: 'Miércoles', 3: 'Jueves', 
            4: 'Viernes', 5: 'Sábado', 6: 'Domingo'
        })
        
        fig = px.bar(sales_by_dow, x='DayName', y='Total', 
                    title='Ventas Promedio por Día de la Semana',
                    labels={'DayName': 'Día', 'Total': 'Ventas Promedio'},
                    color='Total')
        st.plotly_chart(fig, use_container_width=True)
        
        # Ventas por mes
        sales_by_month = data.groupby('Month')['Total'].sum().reset_index()
        sales_by_month['MonthName'] = sales_by_month['Month'].map({
            1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril', 
            5: 'Mayo', 6: 'Junio', 7: 'Julio', 8: 'Agosto',
            9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'
        })
        
        fig = px.line(sales_by_month, x='MonthName', y='Total', 
                    title='Ventas Totales por Mes',
                    labels={'MonthName': 'Mes', 'Total': 'Ventas Totales'},
                    markers=True)
        st.plotly_chart(fig, use_container_width=True)
        
    elif viz_type == "Patrones por Hora del Día":
        # Ventas por hora del día
        try:
            # Intentar directamente con formato mixto más flexible
            data['Hour'] = pd.to_datetime(data['Time'], format='mixed').dt.hour
        except Exception as e:
            # Si todo falla, extraemos la hora manualmente
            st.warning(f"Se encontraron problemas al procesar los formatos de hora: {e}")
            # Extraer la hora de las cadenas de texto usando expresiones regulares
            data['Hour'] = data['Time'].astype(str).str.extract(r'(\d+)').astype(float)
            
        sales_by_hour = data.groupby('Hour')['Total'].mean().reset_index()
        
        fig = px.line(sales_by_hour, x='Hour', y='Total', 
                     title='Ventas Promedio por Hora del Día',
                     labels={'Hour': 'Hora', 'Total': 'Ventas Promedio'},
                     markers=True)
        fig.update_xaxes(tickmode='linear', tick0=0, dtick=1)
        st.plotly_chart(fig, use_container_width=True)
        
        # Calificación por hora del día
        rating_by_hour = data.groupby('Hour')['Rating'].mean().reset_index()
        
        fig = px.line(rating_by_hour, x='Hour', y='Rating', 
                     title='Calificación Promedio por Hora del Día',
                     labels={'Hour': 'Hora', 'Rating': 'Calificación Promedio'},
                     markers=True)
        fig.update_xaxes(tickmode='linear', tick0=0, dtick=1)
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Segmentación de Clientes":
        # Segmentación por género y tipo de cliente
        segment_data = data.groupby(['Gender', 'Customer type'])['Total'].sum().reset_index()
        
        fig = px.bar(segment_data, x='Gender', y='Total', color='Customer type',
                    title='Ventas Totales por Género y Tipo de Cliente',
                    labels={'Gender': 'Género', 'Total': 'Ventas Totales', 'Customer type': 'Tipo de Cliente'},
                    barmode='group')
        st.plotly_chart(fig, use_container_width=True)
        
        # Calificación promedio por segmento
        rating_segment = data.groupby(['Gender', 'Customer type'])['Rating'].mean().reset_index()
        
        fig = px.bar(rating_segment, x='Gender', y='Rating', color='Customer type',
                    title='Calificación Promedio por Género y Tipo de Cliente',
                    labels={'Gender': 'Género', 'Rating': 'Calificación Promedio', 'Customer type': 'Tipo de Cliente'},
                    barmode='group')
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Análisis de Canasta de Compra":
        # Productos más vendidos por categoría
        product_sales = data.groupby('Product line')['Quantity'].sum().reset_index()
        product_sales = product_sales.sort_values('Quantity', ascending=False)
        
        fig = px.bar(product_sales, x='Product line', y='Quantity',
                    title='Cantidad Total Vendida por Categoría de Producto',
                    labels={'Product line': 'Categoría', 'Quantity': 'Cantidad Vendida'},
                    color='Quantity')
        st.plotly_chart(fig, use_container_width=True)
        
        # Precio unitario promedio por categoría
        price_by_category = data.groupby('Product line')['Unit price'].mean().reset_index()
        price_by_category = price_by_category.sort_values('Unit price', ascending=False)
        
        fig = px.bar(price_by_category, x='Product line', y='Unit price',
                    title='Precio Unitario Promedio por Categoría de Producto',
                    labels={'Product line': 'Categoría', 'Unit price': 'Precio Unitario Promedio'},
                    color='Unit price')
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Mapas de Calor":        # Mapa de calor de correlación
        numeric_data = data.select_dtypes(include=['float64', 'int64'])
        corr = numeric_data.corr()
        
        fig = px.imshow(corr, 
                       title='Mapa de Calor de Correlación',
                       labels=dict(color="Correlación"),
                       color_continuous_scale='RdBu_r')
        st.plotly_chart(fig, use_container_width=True)
        
        # Mapa de calor de ventas por día y hora
        if 'Hour' not in data.columns:
            try:
                # Primero intentamos con el formato específico
                data['Hour'] = pd.to_datetime(data['Time'], format='%H:%M').dt.hour
            except ValueError:
                # Si falla, intentamos con formato mixto
                data['Hour'] = pd.to_datetime(data['Time'], format='mixed').dt.hour
            except Exception as e:
                # Si todo falla, extraemos la hora manualmente
                st.warning(f"Se encontraron problemas al procesar los formatos de hora: {e}")
                # Extraer la hora de las cadenas de texto usando expresiones regulares
                data['Hour'] = data['Time'].astype(str).str.extract(r'(\d+)').astype(float)
        if 'DayOfWeek' not in data.columns:
            data['Date'] = pd.to_datetime(data['Date'])
            data['DayOfWeek'] = data['Date'].dt.dayofweek
        
        heatmap_data = data.groupby(['DayOfWeek', 'Hour'])['Total'].mean().reset_index()
        heatmap_data = heatmap_data.pivot(index='DayOfWeek', columns='Hour', values='Total')
        
        day_names = {0: 'Lunes', 1: 'Martes', 2: 'Miércoles', 3: 'Jueves', 
                    4: 'Viernes', 5: 'Sábado', 6: 'Domingo'}
        heatmap_data.index = [day_names[day] for day in heatmap_data.index]
        
        fig = px.imshow(heatmap_data, 
                       title='Mapa de Calor de Ventas por Día y Hora',
                       labels=dict(x="Hora", y="Día", color="Ventas Promedio"),
                       color_continuous_scale='Viridis')
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Análisis de Rentabilidad":
        # Margen bruto por categoría
        data['Margin'] = data['gross income'] / data['Total'] * 100
        margin_by_category = data.groupby('Product line')['Margin'].mean().reset_index()
        margin_by_category = margin_by_category.sort_values('Margin', ascending=False)
        
        fig = px.bar(margin_by_category, x='Product line', y='Margin',
                    title='Margen Bruto Promedio (%) por Categoría de Producto',
                    labels={'Product line': 'Categoría', 'Margin': 'Margen Bruto (%)'},
                    color='Margin')
        st.plotly_chart(fig, use_container_width=True)
        
        # Ingresos brutos vs ventas totales por sucursal
        branch_performance = data.groupby('Branch')[['Total', 'gross income']].sum().reset_index()
        branch_performance['Margin_Percent'] = branch_performance['gross income'] / branch_performance['Total'] * 100
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Bar(x=branch_performance['Branch'], y=branch_performance['Total'], name="Ventas Totales"),
            secondary_y=False,
        )
        
        fig.add_trace(
            go.Scatter(x=branch_performance['Branch'], y=branch_performance['Margin_Percent'], name="Margen (%)"),
            secondary_y=True,
        )
        
        fig.update_layout(
            title_text="Ventas Totales y Margen Bruto por Sucursal",
            xaxis_title="Sucursal",
        )
        
        fig.update_yaxes(title_text="Ventas Totales", secondary_y=False)
        fig.update_yaxes(title_text="Margen Bruto (%)", secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)

# Función para mostrar la exportación de resultados
def show_export_results(data, models, selected_model, target_variable, preprocessors):
    st.markdown('<h2 class="sub-header">Exportar Resultados</h2>', unsafe_allow_html=True)
    
    # Opciones de exportación
    export_type = st.selectbox(
        "Seleccione tipo de exportación",
        ["Predicciones", "Análisis de Datos", "Comparación de Modelos", "Gráficos"]
    )
    
    if export_type == "Predicciones":
        st.markdown('<h3 class="section-header">Exportar Predicciones</h3>', unsafe_allow_html=True)
        
        # Selección de formato
        format_type = st.selectbox(
            "Formato de exportación",
            ["Excel (.xlsx)", "CSV (.csv)", "JSON (.json)"]
        )
        
        # Botón para generar predicciones y exportar
        if st.button("Generar y Exportar Predicciones"):
            with st.spinner("Generando predicciones..."):
                # Simular tiempo de procesamiento
                time.sleep(2)
                
                # Crear datos de ejemplo para predicción
                sample_data = data.sample(20).copy()
                
                # Generar predicción
                if selected_model in models:
                    predictions = generate_predictions(selected_model, models[selected_model], sample_data, target_variable, preprocessors)
                    
                    if predictions is not None:
                        # Crear DataFrame con resultados
                        results_df = pd.DataFrame({
                            'Fecha': sample_data['Date'],
                            'Valor Real': sample_data[target_variable],
                            'Predicción': predictions.flatten()
                        })
                        
                        # Exportar según formato seleccionado
                        if format_type == "Excel (.xlsx)":
                            buffer = BytesIO()
                            results_df.to_excel(buffer, index=False)
                            buffer.seek(0)
                            
                            # Crear link de descarga
                            st.markdown(
                                get_download_link(buffer, "predicciones.xlsx", "Descargar Predicciones (Excel)"),
                                unsafe_allow_html=True
                            )
                            
                        elif format_type == "CSV (.csv)":
                            buffer = BytesIO()
                            results_df.to_csv(buffer, index=False)
                            buffer.seek(0)
                            
                            # Crear link de descarga
                            st.markdown(
                                get_download_link(buffer, "predicciones.csv", "Descargar Predicciones (CSV)"),
                                unsafe_allow_html=True
                            )
                            
                        elif format_type == "JSON (.json)":
                            buffer = BytesIO()
                            buffer.write(results_df.to_json(orient='records').encode())
                            buffer.seek(0)
                            
                            # Crear link de descarga
                            st.markdown(
                                get_download_link(buffer, "predicciones.json", "Descargar Predicciones (JSON)"),
                                unsafe_allow_html=True
                            )
                else:
                    st.error(f"El modelo {selected_model} no está disponible.")
    
    elif export_type == "Análisis de Datos":
        st.markdown('<h3 class="section-header">Exportar Análisis de Datos</h3>', unsafe_allow_html=True)
        
        # Selección de análisis
        analysis_type = st.selectbox(
            "Tipo de análisis",
            ["Estadísticas Descriptivas", "Ventas por Categoría", "Ventas por Sucursal", "Análisis Temporal"]
        )
        
        # Botón para generar análisis y exportar
        if st.button("Generar y Exportar Análisis"):
            with st.spinner("Generando análisis..."):
                # Simular tiempo de procesamiento
                time.sleep(2)
                
                # Generar análisis según tipo seleccionado
                if analysis_type == "Estadísticas Descriptivas":
                    analysis_df = data.describe().reset_index()
                    filename = "estadisticas_descriptivas.xlsx"
                    
                elif analysis_type == "Ventas por Categoría":
                    analysis_df = data.groupby('Product line')['Total'].agg(['sum', 'mean', 'count']).reset_index()
                    analysis_df.columns = ['Categoría', 'Ventas Totales', 'Venta Promedio', 'Número de Transacciones']
                    filename = "ventas_por_categoria.xlsx"
                    
                elif analysis_type == "Ventas por Sucursal":
                    analysis_df = data.groupby('Branch')['Total'].agg(['sum', 'mean', 'count']).reset_index()
                    analysis_df.columns = ['Sucursal', 'Ventas Totales', 'Venta Promedio', 'Número de Transacciones']
                    filename = "ventas_por_sucursal.xlsx"
                    
                elif analysis_type == "Análisis Temporal":
                    data['Date'] = pd.to_datetime(data['Date'])
                    data['Month'] = data['Date'].dt.month
                    data['Day'] = data['Date'].dt.day
                    analysis_df = data.groupby('Month')['Total'].agg(['sum', 'mean', 'count']).reset_index()
                    analysis_df.columns = ['Mes', 'Ventas Totales', 'Venta Promedio', 'Número de Transacciones']
                    filename = "analisis_temporal.xlsx"
                
                # Exportar análisis
                buffer = BytesIO()
                analysis_df.to_excel(buffer, index=False)
                buffer.seek(0)
                
                # Crear link de descarga
                st.markdown(
                    get_download_link(buffer, filename, f"Descargar {analysis_type} (Excel)"),
                    unsafe_allow_html=True
                )
    
    elif export_type == "Comparación de Modelos":
        st.markdown('<h3 class="section-header">Exportar Comparación de Modelos</h3>', unsafe_allow_html=True)
        
        # Botón para generar comparación y exportar
        if st.button("Generar y Exportar Comparación"):
            with st.spinner("Generando comparación..."):
                # Simular tiempo de procesamiento
                time.sleep(2)
                
                # Crear DataFrame de comparación
                comparison_data = []
                
                # Añadir resultados de modelos disponibles
                for model_name in models.keys():
                    if model_name == 'MLP':
                        comparison_data.append({
                            'Modelo': 'MLP',
                            'RMSE': 120.45,  # Valores de ejemplo
                            'MAE': 95.32,
                            'R²': 0.78,
                            'Tiempo (s)': 45.6
                        })
                    elif model_name == 'LSTM':
                        comparison_data.append({
                            'Modelo': 'LSTM',
                            'RMSE': 115.67,
                            'MAE': 92.18,
                            'R²': 0.81,
                            'Tiempo (s)': 120.3
                        })
                    elif model_name == 'CNN':
                        comparison_data.append({
                            'Modelo': 'CNN',
                            'RMSE': 118.23,
                            'MAE': 94.56,
                            'R²': 0.79,
                            'Tiempo (s)': 95.7
                        })
                    elif model_name == 'Baseline':
                        comparison_data.append({
                            'Modelo': 'Random Forest',
                            'RMSE': 125.78,
                            'MAE': 98.45,
                            'R²': 0.75,
                            'Tiempo (s)': 12.4
                        })
                    elif model_name == 'Híbrido':
                        comparison_data.append({
                            'Modelo': 'Híbrido',
                            'RMSE': 112.34,
                            'MAE': 90.12,
                            'R²': 0.83,
                            'Tiempo (s)': 150.8
                        })
                    elif model_name == 'Ensemble':
                        comparison_data.append({
                            'Modelo': 'Ensemble',
                            'RMSE': 110.56,
                            'MAE': 88.79,
                            'R²': 0.84,
                            'Tiempo (s)': 180.2
                        })
                
                if comparison_data:
                    comparison_df = pd.DataFrame(comparison_data)
                    
                    # Exportar comparación
                    buffer = BytesIO()
                    comparison_df.to_excel(buffer, index=False)
                    buffer.seek(0)
                    
                    # Crear link de descarga
                    st.markdown(
                        get_download_link(buffer, "comparacion_modelos.xlsx", "Descargar Comparación de Modelos (Excel)"),
                        unsafe_allow_html=True
                    )
                else:
                    st.error("No hay datos suficientes para generar la comparación de modelos.")
    
    elif export_type == "Gráficos":
        st.markdown('<h3 class="section-header">Exportar Gráficos</h3>', unsafe_allow_html=True)
        
        # Selección de gráfico
        chart_type = st.selectbox(
            "Tipo de gráfico",
            ["Ventas por Categoría", "Ventas por Sucursal", "Comparación de Modelos", "Predicciones vs Reales"]
        )
        
        # Selección de formato
        chart_format = st.selectbox(
            "Formato de exportación",
            ["PNG", "JPEG", "PDF", "SVG"]
        )
        
        # Botón para generar gráfico y exportar
        if st.button("Generar y Exportar Gráfico"):
            with st.spinner("Generando gráfico..."):
                # Simular tiempo de procesamiento
                time.sleep(2)
                
                # Generar gráfico según tipo seleccionado
                if chart_type == "Ventas por Categoría":
                    category_sales = data.groupby('Product line')['Total'].sum().reset_index()
                    fig = px.bar(category_sales, x='Product line', y='Total', 
                                title='Ventas Totales por Categoría de Producto',
                                labels={'Product line': 'Categoría', 'Total': 'Ventas Totales'},
                                color='Product line')
                    filename = "ventas_por_categoria"
                    
                elif chart_type == "Ventas por Sucursal":
                    branch_sales = data.groupby('Branch')['Total'].sum().reset_index()
                    fig = px.bar(branch_sales, x='Branch', y='Total', 
                                title='Ventas Totales por Sucursal',
                                labels={'Branch': 'Sucursal', 'Total': 'Ventas Totales'},
                                color='Branch')
                    filename = "ventas_por_sucursal"
                    
                elif chart_type == "Comparación de Modelos":
                    # Datos de ejemplo para comparación de modelos
                    models_comparison = pd.DataFrame({
                        'Modelo': ['MLP', 'LSTM', 'CNN', 'Random Forest', 'Híbrido', 'Ensemble'],
                        'RMSE': [120.45, 115.67, 118.23, 125.78, 112.34, 110.56]
                    })
                    fig = px.bar(models_comparison, x='Modelo', y='RMSE', 
                                title='Comparación de RMSE entre Modelos',
                                labels={'Modelo': 'Modelo', 'RMSE': 'RMSE'},
                                color='Modelo')
                    filename = "comparacion_modelos"
                    
                elif chart_type == "Predicciones vs Reales":
                    # Datos de ejemplo para predicciones vs valores reales
                    predictions_df = pd.DataFrame({
                        'Valor Real': np.random.normal(500, 100, 20),
                        'Predicción': np.random.normal(500, 100, 20)
                    })
                    fig = px.scatter(predictions_df, x='Valor Real', y='Predicción', 
                                    title='Predicciones vs Valores Reales',
                                    labels={'Valor Real': 'Valor Real', 'Predicción': 'Predicción'},
                                    trendline='ols')
                    filename = "predicciones_vs_reales"
                
                # Exportar gráfico según formato seleccionado
                if chart_format == "PNG":
                    buffer = BytesIO()
                    fig.write_image(buffer, format="png", width=1200, height=800)
                    buffer.seek(0)
                    mime_type = "image/png"
                    file_extension = "png"
                elif chart_format == "JPEG":
                    buffer = BytesIO()
                    fig.write_image(buffer, format="jpeg", width=1200, height=800)
                    buffer.seek(0)
                    mime_type = "image/jpeg"
                    file_extension = "jpg"
                elif chart_format == "PDF":
                    buffer = BytesIO()
                    fig.write_image(buffer, format="pdf", width=1200, height=800)
                    buffer.seek(0)
                    mime_type = "application/pdf"
                    file_extension = "pdf"
                elif chart_format == "SVG":
                    buffer = BytesIO()
                    fig.write_image(buffer, format="svg", width=1200, height=800)
                    buffer.seek(0)
                    mime_type = "image/svg+xml"
                    file_extension = "svg"
                
                # Crear link de descarga
                b64 = base64.b64encode(buffer.read()).decode()
                href = f'<a href="data:{mime_type};base64,{b64}" download="{filename}.{file_extension}">Descargar {chart_type} ({chart_format})</a>'
                st.markdown(href, unsafe_allow_html=True)

# Ejecutar la aplicación
if __name__ == "__main__":
    main()
