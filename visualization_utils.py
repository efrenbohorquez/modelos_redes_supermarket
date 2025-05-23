"""
Utilidades para visualización de datos y resultados de modelos.

Este módulo proporciona funciones para crear visualizaciones avanzadas
para el análisis de datos de ventas de supermercado y la evaluación de
modelos de predicción.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Model
import io
import base64

# Configuración de estilo para matplotlib
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
sns.set_palette('viridis')

def plot_training_history(history, title="Historial de Entrenamiento"):
    """
    Visualiza el historial de entrenamiento de un modelo.
    
    Args:
        history (dict): Historial de entrenamiento del modelo.
        title (str): Título para el gráfico.
    
    Returns:
        fig: Figura de matplotlib.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Gráfico de pérdida
    ax1.plot(history['loss'], label='Entrenamiento')
    ax1.plot(history['val_loss'], label='Validación')
    ax1.set_title('Pérdida durante el Entrenamiento')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Pérdida (MSE)')
    ax1.legend()
    ax1.grid(True)
    
    # Gráfico de MAE
    if 'mae' in history and 'val_mae' in history:
        ax2.plot(history['mae'], label='Entrenamiento')
        ax2.plot(history['val_mae'], label='Validación')
        ax2.set_title('Error Absoluto Medio durante el Entrenamiento')
    else:
        # Usar accuracy si MAE no está disponible
        metric = 'accuracy' if 'accuracy' in history else list(history.keys())[2]
        val_metric = 'val_' + metric
        if val_metric in history:
            ax2.plot(history[metric], label='Entrenamiento')
            ax2.plot(history[val_metric], label='Validación')
            ax2.set_title(f'{metric.capitalize()} durante el Entrenamiento')
    
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Valor')
    ax2.legend()
    ax2.grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    return fig

def plot_predictions(y_true, y_pred, title="Predicciones vs Valores Reales"):
    """
    Visualiza las predicciones vs valores reales.
    
    Args:
        y_true (array): Valores reales.
        y_pred (array): Valores predichos.
        title (str): Título para el gráfico.
    
    Returns:
        fig: Figura de matplotlib.
    """
    fig = plt.figure(figsize=(10, 6))
    
    # Gráfico de dispersión
    plt.scatter(y_true, y_pred, alpha=0.5)
    
    # Línea de referencia (predicción perfecta)
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    # Calcular métricas
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Añadir texto con métricas
    plt.text(
        0.05, 0.95, 
        f'MSE: {mse:.4f}\nRMSE: {rmse:.4f}\nMAE: {mae:.4f}\nR²: {r2:.4f}',
        transform=plt.gca().transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.5)
    )
    
    plt.title(title)
    plt.xlabel('Valores Reales')
    plt.ylabel('Predicciones')
    plt.grid(True)
    plt.tight_layout()
    
    return fig

def plot_feature_importance(model, feature_names, title="Importancia de Características", top_n=15):
    """
    Visualiza la importancia de características para modelos basados en árboles.
    
    Args:
        model: Modelo entrenado (debe tener atributo feature_importances_).
        feature_names (list): Nombres de las características.
        title (str): Título para el gráfico.
        top_n (int): Número de características principales a mostrar.
    
    Returns:
        fig: Figura de matplotlib.
        importance: DataFrame con importancia de características.
    """
    if not hasattr(model, 'feature_importances_'):
        return None, None
    
    # Crear DataFrame con importancia de características
    importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    })
    
    # Ordenar por importancia
    importance = importance.sort_values('Importance', ascending=False).reset_index(drop=True)
    
    # Mostrar top N características
    top_importance = importance.head(top_n)
    
    # Visualizar
    fig = plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=top_importance)
    plt.title(title)
    plt.xlabel('Importancia Relativa')
    plt.ylabel('Característica')
    plt.tight_layout()
    
    return fig, importance

def plot_model_comparison(models_metrics, metric='rmse', title="Comparación de Modelos"):
    """
    Visualiza la comparación de métricas entre diferentes modelos.
    
    Args:
        models_metrics (dict): Diccionario con métricas de modelos.
        metric (str): Métrica a comparar ('rmse', 'mae', 'r2').
        title (str): Título para el gráfico.
    
    Returns:
        fig: Figura de matplotlib.
    """
    # Preparar datos para visualización
    model_names = []
    metric_values = []
    
    for model_name, metrics in models_metrics.items():
        model_names.append(model_name)
        metric_values.append(metrics[metric])
    
    # Crear DataFrame
    df = pd.DataFrame({
        'Modelo': model_names,
        'Valor': metric_values
    })
    
    # Ordenar por valor de métrica (ascendente para rmse/mae, descendente para r2)
    ascending = True if metric.lower() in ['rmse', 'mae'] else False
    df = df.sort_values('Valor', ascending=ascending)
    
    # Visualizar
    fig = plt.figure(figsize=(12, 8))
    
    # Usar colores diferentes según el tipo de modelo
    palette = sns.color_palette("viridis", len(df))
    
    bars = sns.barplot(x='Modelo', y='Valor', data=df, palette=palette)
    
    # Añadir etiquetas con valores
    for i, p in enumerate(bars.patches):
        bars.annotate(f'{p.get_height():.4f}', 
                     (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha = 'center', va = 'bottom',
                     xytext = (0, 5), textcoords = 'offset points')
    
    plt.title(f"{title} - {metric.upper()}")
    plt.xlabel('Modelo')
    plt.ylabel(metric.upper())
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    return fig

def plot_confusion_matrix(cm, class_names, title="Matriz de Confusión"):
    """
    Visualiza una matriz de confusión.
    
    Args:
        cm (array): Matriz de confusión.
        class_names (list): Nombres de las clases.
        title (str): Título para el gráfico.
    
    Returns:
        fig: Figura de matplotlib.
    """
    fig = plt.figure(figsize=(10, 8))
    
    # Normalizar matriz
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Visualizar
    sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues', 
               xticklabels=class_names, yticklabels=class_names)
    
    plt.title(title)
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Etiqueta Predicha')
    plt.tight_layout()
    
    return fig

def plot_time_series(data, date_col, value_col, title="Serie Temporal"):
    """
    Visualiza una serie temporal.
    
    Args:
        data (DataFrame): DataFrame con los datos.
        date_col (str): Nombre de la columna de fecha.
        value_col (str): Nombre de la columna de valor.
        title (str): Título para el gráfico.
    
    Returns:
        fig: Figura de matplotlib.
    """
    fig = plt.figure(figsize=(14, 6))
    
    plt.plot(data[date_col], data[value_col], marker='o', linestyle='-', markersize=4)
    plt.title(title)
    plt.xlabel('Fecha')
    plt.ylabel(value_col)
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig

def plot_time_series_prediction(dates, y_true, y_pred, title="Predicción de Serie Temporal"):
    """
    Visualiza predicciones de series temporales.
    
    Args:
        dates (array): Fechas.
        y_true (array): Valores reales.
        y_pred (array): Valores predichos.
        title (str): Título para el gráfico.
    
    Returns:
        fig: Figura de matplotlib.
    """
    fig = plt.figure(figsize=(14, 6))
    
    plt.plot(dates, y_true, 'b-', label='Real', alpha=0.7)
    plt.plot(dates, y_pred, 'r--', label='Predicción', alpha=0.7)
    
    # Calcular métricas
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Añadir texto con métricas
    plt.text(
        0.05, 0.95, 
        f'MSE: {mse:.4f}\nRMSE: {rmse:.4f}\nMAE: {mae:.4f}\nR²: {r2:.4f}',
        transform=plt.gca().transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.5)
    )
    
    plt.title(title)
    plt.xlabel('Fecha')
    plt.ylabel('Valor')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig

def plot_model_architecture(model, title="Arquitectura del Modelo"):
    """
    Visualiza la arquitectura de un modelo de Keras.
    
    Args:
        model: Modelo de Keras.
        title (str): Título para el gráfico.
    
    Returns:
        dot_img_file: Archivo de imagen con la arquitectura.
    """
    try:
        from tensorflow.keras.utils import plot_model
        import os
        
        # Crear directorio temporal si no existe
        os.makedirs('temp', exist_ok=True)
        
        # Guardar imagen de la arquitectura
        plot_model(model, to_file='temp/model_architecture.png', show_shapes=True, show_layer_names=True)
        
        # Leer imagen
        with open('temp/model_architecture.png', 'rb') as f:
            dot_img_file = f.read()
        
        return dot_img_file
    except Exception as e:
        print(f"Error al visualizar arquitectura: {e}")
        return None

def plot_correlation_matrix(data, title="Matriz de Correlación"):
    """
    Visualiza una matriz de correlación.
    
    Args:
        data (DataFrame): DataFrame con datos numéricos.
        title (str): Título para el gráfico.
    
    Returns:
        fig: Figura de matplotlib.
    """
    # Calcular correlación
    corr = data.corr()
    
    # Generar máscara para el triángulo superior
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # Visualizar
    fig = plt.figure(figsize=(12, 10))
    
    sns.heatmap(corr, mask=mask, cmap='coolwarm', vmax=1, vmin=-1, center=0,
               square=True, linewidths=.5, annot=True, fmt='.2f', cbar_kws={"shrink": .8})
    
    plt.title(title)
    plt.tight_layout()
    
    return fig

def plot_distribution(data, column, title=None):
    """
    Visualiza la distribución de una variable.
    
    Args:
        data (DataFrame): DataFrame con los datos.
        column (str): Nombre de la columna a visualizar.
        title (str): Título para el gráfico.
    
    Returns:
        fig: Figura de matplotlib.
    """
    if title is None:
        title = f"Distribución de {column}"
    
    fig = plt.figure(figsize=(12, 6))
    
    # Histograma con KDE
    sns.histplot(data[column], kde=True)
    
    # Añadir línea vertical para la media
    plt.axvline(data[column].mean(), color='r', linestyle='--', label=f'Media: {data[column].mean():.2f}')
    
    # Añadir línea vertical para la mediana
    plt.axvline(data[column].median(), color='g', linestyle='-.', label=f'Mediana: {data[column].median():.2f}')
    
    plt.title(title)
    plt.xlabel(column)
    plt.ylabel('Frecuencia')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    return fig

def plot_categorical_distribution(data, column, title=None):
    """
    Visualiza la distribución de una variable categórica.
    
    Args:
        data (DataFrame): DataFrame con los datos.
        column (str): Nombre de la columna a visualizar.
        title (str): Título para el gráfico.
    
    Returns:
        fig: Figura de matplotlib.
    """
    if title is None:
        title = f"Distribución de {column}"
    
    # Contar valores
    value_counts = data[column].value_counts().reset_index()
    value_counts.columns = [column, 'Count']
    
    # Calcular porcentajes
    total = value_counts['Count'].sum()
    value_counts['Percentage'] = value_counts['Count'] / total * 100
    
    # Ordenar por conteo
    value_counts = value_counts.sort_values('Count', ascending=False)
    
    fig = plt.figure(figsize=(12, 6))
    
    # Crear gráfico de barras
    ax = sns.barplot(x=column, y='Count', data=value_counts)
    
    # Añadir etiquetas con porcentajes
    for i, p in enumerate(ax.patches):
        percentage = value_counts.iloc[i]['Percentage']
        ax.annotate(f'{percentage:.1f}%', 
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha = 'center', va = 'bottom',
                   xytext = (0, 5), textcoords = 'offset points')
    
    plt.title(title)
    plt.xlabel(column)
    plt.ylabel('Conteo')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y')
    plt.tight_layout()
    
    return fig

def plot_boxplot(data, x, y, title=None):
    """
    Visualiza un boxplot para comparar distribuciones.
    
    Args:
        data (DataFrame): DataFrame con los datos.
        x (str): Nombre de la columna categórica.
        y (str): Nombre de la columna numérica.
        title (str): Título para el gráfico.
    
    Returns:
        fig: Figura de matplotlib.
    """
    if title is None:
        title = f"{y} por {x}"
    
    fig = plt.figure(figsize=(14, 8))
    
    # Crear boxplot
    ax = sns.boxplot(x=x, y=y, data=data)
    
    # Añadir puntos para mostrar distribución
    sns.stripplot(x=x, y=y, data=data, size=4, color=".3", alpha=0.6)
    
    # Añadir medias
    means = data.groupby(x)[y].mean()
    for i, mean_val in enumerate(means):
        ax.annotate(f'Media: {mean_val:.2f}', 
                   xy=(i, mean_val),
                   xytext=(0, 10),
                   textcoords='offset points',
                   ha='center',
                   va='bottom',
                   color='red',
                   fontweight='bold')
    
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y')
    plt.tight_layout()
    
    return fig

def plot_heatmap_calendar(data, date_col, value_col, year=None, title=None):
    """
    Visualiza un mapa de calor tipo calendario.
    
    Args:
        data (DataFrame): DataFrame con los datos.
        date_col (str): Nombre de la columna de fecha.
        value_col (str): Nombre de la columna de valor.
        year (int): Año específico a visualizar.
        title (str): Título para el gráfico.
    
    Returns:
        fig: Figura de matplotlib.
    """
    # Asegurarse de que la columna de fecha sea datetime
    data = data.copy()
    data[date_col] = pd.to_datetime(data[date_col])
    
    # Filtrar por año si se especifica
    if year is not None:
        data = data[data[date_col].dt.year == year]
    
    # Extraer año, mes y día
    data['Year'] = data[date_col].dt.year
    data['Month'] = data[date_col].dt.month
    data['Day'] = data[date_col].dt.day
    
    # Agrupar por año, mes y día
    heatmap_data = data.groupby(['Year', 'Month', 'Day'])[value_col].mean().reset_index()
    
    # Crear pivote para el mapa de calor
    pivot_data = heatmap_data.pivot_table(index='Day', columns=['Year', 'Month'], values=value_col)
    
    if title is None:
        title = f"Mapa de Calor de {value_col} por Fecha"
    
    fig = plt.figure(figsize=(16, 10))
    
    # Crear mapa de calor
    sns.heatmap(pivot_data, cmap='viridis', linewidths=.5, annot=False, cbar_kws={"shrink": .8})
    
    # Crear etiquetas de mes-año para las columnas
    month_names = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
    col_labels = []
    for year, month in pivot_data.columns:
        col_labels.append(f"{month_names[month-1]}-{year}")
    
    plt.xticks(np.arange(len(col_labels)) + 0.5, col_labels, rotation=90)
    plt.title(title)
    plt.tight_layout()
    
    return fig

def fig_to_base64(fig):
    """
    Convierte una figura de matplotlib a base64 para incrustar en HTML.
    
    Args:
        fig: Figura de matplotlib.
    
    Returns:
        str: String en formato base64.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    return img_str

def create_plotly_time_series(data, date_col, value_col, title="Serie Temporal"):
    """
    Crea un gráfico interactivo de serie temporal con Plotly.
    
    Args:
        data (DataFrame): DataFrame con los datos.
        date_col (str): Nombre de la columna de fecha.
        value_col (str): Nombre de la columna de valor.
        title (str): Título para el gráfico.
    
    Returns:
        fig: Figura de Plotly.
    """
    fig = px.line(data, x=date_col, y=value_col, 
                 title=title,
                 labels={date_col: 'Fecha', value_col: 'Valor'},
                 line_shape='linear')
    
    fig.update_layout(
        xaxis_title='Fecha',
        yaxis_title=value_col,
        hovermode='x unified',
        height=500
    )
    
    return fig

def create_plotly_bar_chart(data, x, y, title, color=None):
    """
    Crea un gráfico de barras interactivo con Plotly.
    
    Args:
        data (DataFrame): DataFrame con los datos.
        x (str): Nombre de la columna para el eje X.
        y (str): Nombre de la columna para el eje Y.
        title (str): Título para el gráfico.
        color (str): Nombre de la columna para color.
    
    Returns:
        fig: Figura de Plotly.
    """
    fig = px.bar(data, x=x, y=y, 
                title=title,
                labels={x: x, y: y},
                color=color if color else x)
    
    fig.update_layout(
        xaxis_title=x,
        yaxis_title=y,
        height=500
    )
    
    return fig

def create_plotly_scatter(data, x, y, title, color=None, size=None, hover_data=None):
    """
    Crea un gráfico de dispersión interactivo con Plotly.
    
    Args:
        data (DataFrame): DataFrame con los datos.
        x (str): Nombre de la columna para el eje X.
        y (str): Nombre de la columna para el eje Y.
        title (str): Título para el gráfico.
        color (str): Nombre de la columna para color.
        size (str): Nombre de la columna para tamaño.
        hover_data (list): Lista de columnas para mostrar en hover.
    
    Returns:
        fig: Figura de Plotly.
    """
    fig = px.scatter(data, x=x, y=y, 
                    title=title,
                    labels={x: x, y: y},
                    color=color,
                    size=size,
                    hover_data=hover_data)
    
    fig.update_layout(
        xaxis_title=x,
        yaxis_title=y,
        height=500
    )
    
    return fig

def create_plotly_heatmap(data, title="Mapa de Calor"):
    """
    Crea un mapa de calor interactivo con Plotly.
    
    Args:
        data (DataFrame/array): Datos para el mapa de calor.
        title (str): Título para el gráfico.
    
    Returns:
        fig: Figura de Plotly.
    """
    fig = px.imshow(data, 
                   title=title,
                   labels=dict(color="Valor"),
                   color_continuous_scale='Viridis')
    
    fig.update_layout(
        height=600
    )
    
    return fig

def create_plotly_pie_chart(data, names, values, title="Gráfico Circular"):
    """
    Crea un gráfico circular interactivo con Plotly.
    
    Args:
        data (DataFrame): DataFrame con los datos.
        names (str): Nombre de la columna para etiquetas.
        values (str): Nombre de la columna para valores.
        title (str): Título para el gráfico.
    
    Returns:
        fig: Figura de Plotly.
    """
    fig = px.pie(data, names=names, values=values, 
                title=title)
    
    fig.update_layout(
        height=500
    )
    
    return fig

def create_plotly_model_comparison(models_metrics, metric='rmse', title="Comparación de Modelos"):
    """
    Crea un gráfico de barras interactivo para comparar modelos con Plotly.
    
    Args:
        models_metrics (dict): Diccionario con métricas de modelos.
        metric (str): Métrica a comparar ('rmse', 'mae', 'r2').
        title (str): Título para el gráfico.
    
    Returns:
        fig: Figura de Plotly.
    """
    # Preparar datos para visualización
    model_names = []
    metric_values = []
    
    for model_name, metrics in models_metrics.items():
        model_names.append(model_name)
        metric_values.append(metrics[metric])
    
    # Crear DataFrame
    df = pd.DataFrame({
        'Modelo': model_names,
        'Valor': metric_values
    })
    
    # Ordenar por valor de métrica (ascendente para rmse/mae, descendente para r2)
    ascending = True if metric.lower() in ['rmse', 'mae'] else False
    df = df.sort_values('Valor', ascending=ascending)
    
    # Crear gráfico
    fig = px.bar(df, x='Modelo', y='Valor', 
                title=f"{title} - {metric.upper()}",
                color='Modelo')
    
    fig.update_layout(
        xaxis_title='Modelo',
        yaxis_title=metric.upper(),
        height=500
    )
    
    return fig
