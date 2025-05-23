"""
Utilidades para la documentación y descripción de modelos.

Este módulo proporciona funciones para generar descripciones detalladas
de los modelos implementados, sus parámetros, arquitecturas y resultados,
facilitando la documentación académica y la publicación web.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
import json
import os

def get_model_summary(model, line_length=100):
    """
    Obtiene un resumen detallado de la arquitectura del modelo.
    
    Args:
        model: Modelo de Keras o scikit-learn.
        line_length (int): Longitud máxima de línea para el resumen.
    
    Returns:
        str: Resumen del modelo.
    """
    # Para modelos de Keras
    if isinstance(model, tf.keras.Model):
        stringlist = []
        model.summary(print_fn=lambda x: stringlist.append(x), line_length=line_length)
        summary = "\n".join(stringlist)
        return summary
    
    # Para modelos de scikit-learn
    else:
        try:
            return str(model)
        except:
            return "No se pudo obtener el resumen del modelo."

def get_model_parameters(model):
    """
    Obtiene los parámetros del modelo.
    
    Args:
        model: Modelo de Keras o scikit-learn.
    
    Returns:
        dict: Diccionario con los parámetros del modelo.
    """
    # Para modelos de Keras
    if isinstance(model, tf.keras.Model):
        params = {}
        
        # Obtener optimizador y tasa de aprendizaje
        try:
            optimizer = model.optimizer
            params['optimizer'] = optimizer.__class__.__name__
            
            if hasattr(optimizer, 'lr'):
                params['learning_rate'] = float(optimizer.lr.numpy())
            elif hasattr(optimizer, 'learning_rate'):
                params['learning_rate'] = float(optimizer.learning_rate.numpy())
        except:
            params['optimizer'] = "Desconocido"
            params['learning_rate'] = "Desconocido"
        
        # Obtener función de pérdida y métricas
        try:
            params['loss'] = model.loss
            params['metrics'] = [m.__name__ if callable(m) else m for m in model.metrics]
        except:
            params['loss'] = "Desconocido"
            params['metrics'] = "Desconocido"
        
        # Contar parámetros totales
        params['total_params'] = model.count_params()
        
        return params
    
    # Para modelos de scikit-learn
    else:
        try:
            return model.get_params()
        except:
            return {"message": "No se pudieron obtener los parámetros del modelo."}

def get_model_layers_info(model):
    """
    Obtiene información detallada de las capas del modelo.
    
    Args:
        model: Modelo de Keras.
    
    Returns:
        list: Lista de diccionarios con información de cada capa.
    """
    if not isinstance(model, tf.keras.Model):
        return [{"message": "El modelo no es una instancia de Keras Model."}]
    
    layers_info = []
    
    for i, layer in enumerate(model.layers):
        layer_info = {
            'index': i,
            'name': layer.name,
            'type': layer.__class__.__name__,
            'output_shape': str(layer.output_shape),
            'params': layer.count_params()
        }
        
        # Obtener configuración específica según el tipo de capa
        config = {}
        
        if isinstance(layer, tf.keras.layers.Dense):
            config['units'] = layer.units
            config['activation'] = layer.activation.__name__ if layer.activation else 'None'
            
        elif isinstance(layer, tf.keras.layers.Conv2D):
            config['filters'] = layer.filters
            config['kernel_size'] = layer.kernel_size
            config['strides'] = layer.strides
            config['padding'] = layer.padding
            config['activation'] = layer.activation.__name__ if layer.activation else 'None'
            
        elif isinstance(layer, tf.keras.layers.LSTM):
            config['units'] = layer.units
            config['return_sequences'] = layer.return_sequences
            config['activation'] = layer.activation.__name__ if layer.activation else 'None'
            
        elif isinstance(layer, tf.keras.layers.Dropout):
            config['rate'] = layer.rate
            
        elif isinstance(layer, tf.keras.layers.BatchNormalization):
            config['axis'] = layer.axis
            config['momentum'] = layer.momentum
            
        layer_info['config'] = config
        layers_info.append(layer_info)
    
    return layers_info

def generate_model_description(model_name, model_type="Red Neuronal"):
    """
    Genera una descripción detallada del modelo.
    
    Args:
        model_name (str): Nombre del modelo.
        model_type (str): Tipo de modelo.
    
    Returns:
        str: Descripción detallada del modelo.
    """
    descriptions = {
        "MLP": {
            "title": "Red Neuronal Multicapa (MLP)",
            "description": """
            La Red Neuronal Multicapa (MLP) es un tipo de red neuronal artificial compuesta por múltiples capas de neuronas. 
            En este proyecto, se utiliza para predecir ventas basándose en características tabulares.
            
            Una MLP consiste en al menos tres capas: una capa de entrada, una o más capas ocultas y una capa de salida. 
            Cada neurona en una capa está conectada a todas las neuronas de la capa siguiente, formando una red completamente conectada.
            """,
            "architecture": """
            La arquitectura implementada consta de:
            - Capa de entrada: Recibe las características preprocesadas
            - Capas ocultas: Múltiples capas densas con activación ReLU
            - Regularización: Dropout y BatchNormalization para prevenir sobreajuste
            - Capa de salida: Una neurona con activación lineal para regresión
            
            El modelo utiliza el optimizador Adam con tasa de aprendizaje adaptativa y función de pérdida MSE (Error Cuadrático Medio).
            """,
            "advantages": [
                "Capacidad para modelar relaciones no lineales complejas",
                "Buen rendimiento en datos tabulares estructurados",
                "Relativamente rápido de entrenar comparado con otras arquitecturas de redes neuronales",
                "Flexibilidad para ajustar la complejidad del modelo mediante el número de capas y neuronas"
            ],
            "limitations": [
                "No captura dependencias temporales explícitamente",
                "Puede sobreajustarse en conjuntos de datos pequeños",
                "Requiere normalización de datos para un rendimiento óptimo",
                "Sensible a la inicialización de pesos y otros hiperparámetros"
            ],
            "use_cases": [
                "Predicción de ventas totales basada en características del cliente y producto",
                "Estimación de calificaciones de clientes",
                "Predicción de ingresos brutos",
                "Análisis de factores que influyen en las ventas"
            ]
        },
        "LSTM": {
            "title": "Red Neuronal Recurrente LSTM",
            "description": """
            Las redes LSTM (Long Short-Term Memory) son un tipo de red neuronal recurrente diseñada para capturar dependencias temporales a largo plazo en secuencias de datos.
            
            A diferencia de las redes neuronales tradicionales, las LSTM tienen conexiones de retroalimentación que les permiten mantener información a lo largo del tiempo, 
            haciéndolas ideales para el análisis de series temporales y datos secuenciales.
            """,
            "architecture": """
            La arquitectura implementada consta de:
            - Capa de entrada: Recibe secuencias de datos con forma [muestras, pasos_tiempo, características]
            - Capas LSTM: Celdas con mecanismos de compuerta (input, forget, output) que controlan el flujo de información
            - Regularización: Dropout y BatchNormalization para prevenir sobreajuste
            - Capa de salida: Una neurona con activación lineal para regresión
            
            El modelo utiliza ventanas deslizantes para crear secuencias temporales a partir de los datos históricos.
            """,
            "advantages": [
                "Excelente para capturar patrones temporales y estacionales",
                "Maneja eficientemente dependencias a largo plazo",
                "Ideal para predicción de series temporales",
                "Capacidad para recordar información relevante y olvidar información irrelevante"
            ],
            "limitations": [
                "Requiere datos secuenciales estructurados",
                "Mayor tiempo de entrenamiento comparado con modelos más simples",
                "Necesita suficientes datos históricos para ser efectivo",
                "Mayor complejidad computacional"
            ],
            "use_cases": [
                "Predicción de ventas diarias basada en patrones históricos",
                "Análisis de tendencias temporales en comportamiento de clientes",
                "Detección de patrones estacionales en ventas",
                "Predicción de demanda futura de productos"
            ]
        },
        "CNN": {
            "title": "Red Neuronal Convolucional (CNN)",
            "description": """
            Aunque tradicionalmente utilizadas para procesamiento de imágenes, las CNN pueden adaptarse para datos tabulares y series temporales 
            mediante la conversión de datos a formatos matriciales.
            
            Las CNN utilizan operaciones de convolución para detectar patrones locales en los datos, lo que les permite capturar 
            características espaciales y temporales de manera eficiente.
            """,
            "architecture": """
            La arquitectura implementada consta de:
            - Capa de entrada: Recibe datos transformados en formato matricial [muestras, altura, anchura, canales]
            - Capas convolucionales: Filtros que detectan patrones locales
            - Capas de pooling: Reducción de dimensionalidad para capturar características importantes
            - Regularización: Dropout y BatchNormalization para prevenir sobreajuste
            - Capa de aplanamiento: Convierte mapas de características en vector
            - Capas densas: Procesamiento final de características extraídas
            - Capa de salida: Una neurona con activación lineal para regresión
            """,
            "advantages": [
                "Capacidad para detectar patrones locales en los datos",
                "Reducción automática de dimensionalidad",
                "Robustez ante variaciones en los datos",
                "Extracción jerárquica de características"
            ],
            "limitations": [
                "Requiere transformación de datos tabulares a formato matricial",
                "Puede ser complejo de interpretar",
                "Mayor número de hiperparámetros a ajustar",
                "Potencialmente mayor tiempo de entrenamiento"
            ],
            "use_cases": [
                "Detección de patrones complejos en datos de ventas",
                "Análisis de relaciones espaciales entre variables",
                "Identificación de patrones de compra no evidentes",
                "Procesamiento de datos multidimensionales"
            ]
        },
        "Baseline": {
            "title": "Modelos Baseline (Random Forest)",
            "description": """
            Los modelos baseline proporcionan un punto de referencia para evaluar el rendimiento de modelos más complejos. 
            En este proyecto, se utiliza Random Forest como modelo baseline principal.
            
            Random Forest es un algoritmo de ensemble que combina múltiples árboles de decisión para mejorar la precisión 
            y controlar el sobreajuste, siendo muy efectivo para problemas de regresión y clasificación.
            """,
            "architecture": """
            La arquitectura implementada consta de:
            - Conjunto de árboles de decisión entrenados con diferentes subconjuntos de datos (bagging)
            - Selección aleatoria de características en cada nodo
            - Promedio de predicciones de todos los árboles para obtener la predicción final
            
            El modelo ha sido optimizado mediante búsqueda de hiperparámetros para encontrar la mejor configuración.
            """,
            "advantages": [
                "Robusto ante outliers y ruido",
                "Proporciona importancia de características",
                "No requiere escalado de datos",
                "Rápido de entrenar y predecir",
                "Menos propenso a sobreajuste que un solo árbol de decisión"
            ],
            "limitations": [
                "Puede tener dificultades con relaciones muy complejas",
                "Menos efectivo en datos altamente dimensionales",
                "Tendencia a sobreajustar si no se controlan hiperparámetros",
                "Modelo de caja negra con interpretabilidad limitada"
            ],
            "use_cases": [
                "Predicción rápida de ventas con buena precisión",
                "Identificación de factores importantes que influyen en las ventas",
                "Análisis de segmentos de clientes",
                "Punto de referencia para evaluar modelos más complejos"
            ]
        },
        "Híbrido": {
            "title": "Modelo Híbrido (MLP+LSTM+CNN)",
            "description": """
            El modelo híbrido combina diferentes arquitecturas de redes neuronales para aprovechar las fortalezas de cada una 
            y compensar sus debilidades.
            
            Este enfoque permite capturar simultáneamente diferentes tipos de patrones en los datos: relaciones no lineales (MLP), 
            dependencias temporales (LSTM) y patrones locales (CNN).
            """,
            "architecture": """
            La arquitectura implementada consta de tres ramas paralelas:
            - Rama MLP: Procesa características tabulares
            - Rama LSTM: Procesa secuencias temporales
            - Rama CNN: Procesa datos en formato matricial
            
            Las salidas de las tres ramas se concatenan y se procesan a través de capas densas adicionales 
            antes de la predicción final.
            """,
            "advantages": [
                "Captura simultáneamente patrones tabulares, temporales y espaciales",
                "Mayor capacidad expresiva que modelos individuales",
                "Potencialmente mejor rendimiento en problemas complejos",
                "Flexibilidad para adaptarse a diferentes tipos de datos"
            ],
            "limitations": [
                "Mayor complejidad computacional",
                "Requiere más datos para entrenar efectivamente",
                "Mayor riesgo de sobreajuste si no se regulariza adecuadamente",
                "Más difícil de implementar y mantener"
            ],
            "use_cases": [
                "Predicción de ventas considerando múltiples factores y patrones",
                "Análisis holístico del comportamiento de clientes",
                "Predicción de tendencias complejas",
                "Cuando se requiere máxima precisión"
            ]
        },
        "Ensemble": {
            "title": "Modelo Ensemble",
            "description": """
            Los modelos ensemble combinan las predicciones de múltiples modelos para obtener predicciones más robustas y precisas.
            
            En este proyecto, se implementan diferentes técnicas de ensemble: Voting (promedio de predicciones), 
            Stacking (meta-modelo que aprende a combinar predicciones) y ensemble manual.
            """,
            "architecture": """
            Las arquitecturas implementadas incluyen:
            - Voting Regressor: Promedio de predicciones de múltiples modelos base
            - Stacking Regressor: Meta-modelo que aprende a combinar predicciones de modelos base
            - Ensemble Manual: Combinación ponderada de predicciones de diferentes modelos
            
            Los modelos base incluyen tanto algoritmos tradicionales (Random Forest, Gradient Boosting) 
            como redes neuronales (MLP, LSTM, CNN).
            """,
            "advantages": [
                "Reduce varianza y sesgo",
                "Mayor robustez ante diferentes condiciones de datos",
                "Generalmente mejor rendimiento que modelos individuales",
                "Menos sensible a outliers y ruido"
            ],
            "limitations": [
                "Mayor complejidad computacional",
                "Más difícil de interpretar",
                "Requiere gestión de múltiples modelos",
                "Potencial overhead en tiempo de predicción"
            ],
            "use_cases": [
                "Predicción de ventas con máxima precisión y robustez",
                "Cuando se requiere alta confiabilidad en las predicciones",
                "Combinación de diferentes perspectivas de análisis",
                "Reducción de riesgo de predicciones erróneas"
            ]
        }
    }
    
    # Obtener descripción según el modelo
    if model_name in descriptions:
        desc = descriptions[model_name]
        
        # Formatear ventajas y limitaciones como listas
        advantages_list = "\n".join([f"- {adv}" for adv in desc["advantages"]])
        limitations_list = "\n".join([f"- {lim}" for lim in desc["limitations"]])
        use_cases_list = "\n".join([f"- {uc}" for uc in desc["use_cases"]])
        
        # Construir descripción completa
        full_description = f"""
        # {desc["title"]}
        
        ## Descripción
        {desc["description"]}
        
        ## Arquitectura
        {desc["architecture"]}
        
        ## Ventajas
        {advantages_list}
        
        ## Limitaciones
        {limitations_list}
        
        ## Casos de Uso
        {use_cases_list}
        """
        
        return full_description
    else:
        # Descripción genérica si no se encuentra el modelo específico
        return f"""
        # {model_name}
        
        ## Descripción
        {model_name} es un tipo de {model_type} implementado en este proyecto para la predicción de ventas de supermercado.
        
        Para obtener más detalles sobre este modelo, consulte la documentación específica del modelo.
        """

def generate_model_comparison_table(models_metrics):
    """
    Genera una tabla de comparación de modelos.
    
    Args:
        models_metrics (dict): Diccionario con métricas de modelos.
    
    Returns:
        pd.DataFrame: DataFrame con la comparación de modelos.
    """
    comparison_data = []
    
    for model_name, metrics in models_metrics.items():
        row = {
            'Modelo': model_name,
            'RMSE': metrics.get('rmse', None),
            'MAE': metrics.get('mae', None),
            'R²': metrics.get('r2', None),
            'Tiempo (s)': metrics.get('training_time', None)
        }
        comparison_data.append(row)
    
    # Crear DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    
    # Ordenar por RMSE (ascendente)
    if 'RMSE' in comparison_df.columns:
        comparison_df = comparison_df.sort_values('RMSE')
    
    return comparison_df

def generate_model_recommendations(comparison_df):
    """
    Genera recomendaciones basadas en la comparación de modelos.
    
    Args:
        comparison_df (pd.DataFrame): DataFrame con la comparación de modelos.
    
    Returns:
        str: Texto con recomendaciones.
    """
    recommendations = """
    # Recomendaciones para Propietarios de Supermercados
    
    Basado en el análisis comparativo de los diferentes modelos de predicción, ofrecemos las siguientes recomendaciones:
    """
    
    # Si hay datos de comparación
    if not comparison_df.empty:
        # Mejor modelo por precisión (RMSE)
        if 'RMSE' in comparison_df.columns:
            best_rmse_model = comparison_df.iloc[0]['Modelo']
            recommendations += f"""
            ## Para máxima precisión
            
            El modelo **{best_rmse_model}** ofrece la mayor precisión en términos de RMSE, lo que lo hace ideal para:
            - Predicciones críticas de ventas que requieren alta precisión
            - Planificación de inventario a largo plazo
            - Decisiones estratégicas basadas en pronósticos de ventas
            """
        
        # Modelo más rápido
        if 'Tiempo (s)' in comparison_df.columns:
            fastest_model = comparison_df.sort_values('Tiempo (s)').iloc[0]['Modelo']
            recommendations += f"""
            ## Para análisis rápidos
            
            El modelo **{fastest_model}** ofrece el mejor equilibrio entre velocidad y precisión, lo que lo hace ideal para:
            - Análisis en tiempo real o con restricciones de tiempo
            - Pruebas rápidas de diferentes escenarios
            - Implementación en sistemas con recursos limitados
            """
        
        # Mejor modelo por interpretabilidad
        interpretable_models = ['Random Forest', 'Linear Regression', 'Decision Tree', 'Baseline']
        interpretable_in_df = [model for model in interpretable_models if model in comparison_df['Modelo'].values]
        
        if interpretable_in_df:
            best_interpretable = comparison_df[comparison_df['Modelo'].isin(interpretable_in_df)].iloc[0]['Modelo']
            recommendations += f"""
            ## Para interpretabilidad
            
            El modelo **{best_interpretable}** ofrece el mejor equilibrio entre precisión e interpretabilidad, lo que lo hace ideal para:
            - Entender los factores que influyen en las ventas
            - Comunicar resultados a stakeholders no técnicos
            - Tomar decisiones basadas en factores específicos
            """
    
    # Recomendaciones generales
    recommendations += """
    ## Recomendaciones generales
    
    1. **Para predicciones diarias de ventas totales**: Utilice el modelo Ensemble o Híbrido, que generalmente ofrecen el mejor equilibrio entre precisión y robustez.
    
    2. **Para análisis rápidos o con recursos limitados**: El modelo Random Forest optimizado proporciona un buen rendimiento con menor complejidad computacional.
    
    3. **Para predicciones con patrones estacionales fuertes**: Priorice el modelo LSTM, que captura eficientemente dependencias temporales.
    
    4. **Para entender factores que influyen en las ventas**: Utilice Random Forest, que proporciona importancia de características interpretable.
    
    5. **Para predicciones a largo plazo**: Combine los resultados de múltiples modelos (Ensemble) para obtener mayor robustez.
    
    6. **Para implementación en sistemas con recursos limitados**: Considere el modelo MLP o Random Forest, que ofrecen buen equilibrio entre rendimiento y eficiencia.
    """
    
    return recommendations

def generate_academic_report_structure():
    """
    Genera la estructura para un informe académico.
    
    Returns:
        dict: Diccionario con la estructura del informe.
    """
    structure = {
        "title": "Análisis y Predicción de Ventas de Supermercado mediante Redes Neuronales y Técnicas de Aprendizaje Automático",
        "sections": [
            {
                "title": "Resumen",
                "content": """
                Este estudio presenta un análisis exhaustivo de datos de ventas de supermercado y el desarrollo de modelos predictivos 
                basados en redes neuronales y técnicas de aprendizaje automático. Se implementaron y evaluaron seis modelos diferentes: 
                Red Neuronal Multicapa (MLP), Red Neuronal Recurrente LSTM, Red Neuronal Convolucional (CNN), modelos baseline, 
                un modelo híbrido y un ensemble de modelos. Los resultados muestran que [completar con hallazgos principales]. 
                Este trabajo proporciona a los propietarios de pequeños supermercados herramientas para la toma de decisiones 
                basada en datos, permitiéndoles optimizar inventario, personal y estrategias de marketing.
                """
            },
            {
                "title": "Introducción",
                "content": """
                La predicción precisa de ventas es fundamental para la gestión eficiente de supermercados, especialmente para 
                pequeños negocios con recursos limitados. Este estudio aborda esta necesidad mediante la aplicación de técnicas 
                avanzadas de aprendizaje automático y redes neuronales para analizar y predecir patrones de ventas.
                
                ### Antecedentes
                
                Los supermercados generan grandes cantidades de datos transaccionales que contienen información valiosa sobre 
                patrones de compra, preferencias de clientes y tendencias de ventas. Sin embargo, extraer conocimiento útil 
                de estos datos requiere técnicas analíticas avanzadas.
                
                ### Objetivos
                
                1. Analizar patrones y tendencias en datos históricos de ventas de supermercado
                2. Desarrollar y comparar múltiples modelos predictivos basados en diferentes arquitecturas
                3. Identificar factores clave que influyen en las ventas
                4. Proporcionar recomendaciones prácticas para propietarios de pequeños supermercados
                
                ### Relevancia
                
                Este estudio contribuye tanto al campo académico del análisis de datos y aprendizaje automático como a la 
                práctica empresarial en el sector minorista, demostrando la aplicabilidad de técnicas avanzadas en contextos 
                de negocio reales.
                """
            },
            {
                "title": "Marco Teórico",
                "content": """
                ### Redes Neuronales Artificiales
                
                Las redes neuronales artificiales son modelos computacionales inspirados en el funcionamiento del cerebro humano, 
                compuestos por unidades interconectadas (neuronas) organizadas en capas. Han demostrado gran efectividad en 
                problemas de predicción y clasificación complejos.
                
                #### Red Neuronal Multicapa (MLP)
                
                La MLP es una red neuronal feedforward compuesta por múltiples capas de neuronas. Cada neurona utiliza una 
                función de activación no lineal, permitiendo al modelo aprender relaciones complejas entre variables.
                
                #### Red Neuronal Recurrente LSTM
                
                Las redes LSTM (Long Short-Term Memory) son un tipo de red neuronal recurrente diseñada para capturar dependencias 
                temporales a largo plazo, siendo particularmente efectivas para datos secuenciales y series temporales.
                
                #### Red Neuronal Convolucional (CNN)
                
                Aunque tradicionalmente utilizadas para procesamiento de imágenes, las CNN pueden adaptarse para datos tabulares 
                mediante la transformación adecuada, permitiendo detectar patrones locales en los datos.
                
                ### Modelos de Ensemble
                
                Los modelos de ensemble combinan múltiples modelos para mejorar la precisión y robustez de las predicciones, 
                reduciendo la varianza y el sesgo inherentes a modelos individuales.
                
                ### Métricas de Evaluación
                
                Para evaluar el rendimiento de los modelos se utilizan métricas como RMSE (Root Mean Squared Error), 
                MAE (Mean Absolute Error) y R² (Coeficiente de Determinación), cada una proporcionando diferentes perspectivas 
                sobre la calidad de las predicciones.
                """
            },
            {
                "title": "Metodología",
                "content": """
                ### Datos
                
                Se utilizó un conjunto de datos de ventas de supermercado que incluye información sobre transacciones, 
                productos, clientes y sucursales. Los datos abarcan [completar con período] y contienen [completar con número] 
                transacciones de [completar con número] sucursales.
                
                ### Preprocesamiento de Datos
                
                El preprocesamiento incluyó:
                - Limpieza de datos (manejo de valores faltantes y outliers)
                - Transformación de variables categóricas
                - Normalización de variables numéricas
                - Creación de características adicionales
                - Preparación de datos específica para cada tipo de modelo
                
                ### Implementación de Modelos
                
                Se implementaron seis modelos diferentes:
                1. Red Neuronal Multicapa (MLP)
                2. Red Neuronal Recurrente LSTM
                3. Red Neuronal Convolucional (CNN)
                4. Modelos Baseline (Random Forest, Regresión)
                5. Modelo Híbrido (combinación de MLP, LSTM y CNN)
                6. Modelo Ensemble (combinación de múltiples modelos)
                
                ### Evaluación y Validación
                
                Los modelos fueron evaluados utilizando:
                - Validación cruzada
                - Conjunto de prueba independiente
                - Métricas de rendimiento: RMSE, MAE, R²
                - Análisis de importancia de características
                """
            },
            {
                "title": "Resultados",
                "content": """
                ### Análisis Exploratorio
                
                El análisis exploratorio reveló [completar con hallazgos principales], incluyendo patrones de ventas por 
                categoría de producto, sucursal, día de la semana y hora del día.
                
                ### Rendimiento de Modelos
                
                [Completar con tabla comparativa de rendimiento de modelos]
                
                El modelo [completar con mejor modelo] mostró el mejor rendimiento con un RMSE de [completar con valor], 
                seguido por [completar con segundo mejor modelo] con un RMSE de [completar con valor].
                
                ### Análisis de Características
                
                Las características más influyentes en las predicciones fueron [completar con características importantes], 
                lo que sugiere que [completar con interpretación].
                
                ### Visualizaciones
                
                [Completar con descripción de visualizaciones clave]
                """
            },
            {
                "title": "Discusión",
                "content": """
                ### Interpretación de Resultados
                
                Los resultados indican que [completar con interpretación general], lo que concuerda con [completar con literatura 
                o conocimiento previo]. La superioridad de [completar con mejor modelo] puede atribuirse a [completar con razones].
                
                ### Implicaciones Prácticas
                
                Para los propietarios de pequeños supermercados, estos hallazgos sugieren varias estrategias:
                - [Completar con recomendación 1]
                - [Completar con recomendación 2]
                - [Completar con recomendación 3]
                
                ### Limitaciones
                
                Este estudio presenta algunas limitaciones:
                - [Completar con limitación 1]
                - [Completar con limitación 2]
                - [Completar con limitación 3]
                
                ### Trabajo Futuro
                
                Futuras investigaciones podrían:
                - [Completar con dirección futura 1]
                - [Completar con dirección futura 2]
                - [Completar con dirección futura 3]
                """
            },
            {
                "title": "Conclusiones",
                "content": """
                Este estudio demuestra la efectividad de las redes neuronales y técnicas de aprendizaje automático para 
                la predicción de ventas en supermercados. Los resultados indican que [completar con conclusión principal].
                
                La implementación de estos modelos en un sistema integrado proporciona a los propietarios de pequeños 
                supermercados herramientas valiosas para la toma de decisiones basada en datos, permitiéndoles optimizar 
                operaciones y mejorar la rentabilidad.
                
                Las metodologías y hallazgos presentados contribuyen tanto al campo académico como a la práctica empresarial, 
                demostrando cómo las técnicas avanzadas de análisis de datos pueden aplicarse efectivamente en contextos 
                de negocio reales.
                """
            },
            {
                "title": "Referencias",
                "content": """
                [Completar con lista de referencias en formato APA]
                """
            },
            {
                "title": "Apéndices",
                "content": """
                ### Apéndice A: Detalles de Implementación
                
                [Completar con detalles técnicos adicionales]
                
                ### Apéndice B: Resultados Adicionales
                
                [Completar con resultados adicionales]
                
                ### Apéndice C: Código Fuente
                
                [Completar con información sobre acceso al código fuente]
                """
            }
        ]
    }
    
    return structure

def generate_web_publication_structure():
    """
    Genera la estructura para una publicación web dirigida a propietarios de supermercados.
    
    Returns:
        dict: Diccionario con la estructura de la publicación web.
    """
    structure = {
        "title": "Optimización de Ventas para Pequeños Supermercados: Herramientas de Predicción y Análisis",
        "sections": [
            {
                "title": "Introducción",
                "content": """
                En el competitivo mercado minorista actual, los pequeños supermercados enfrentan desafíos únicos para mantenerse 
                rentables y competitivos. Este proyecto proporciona herramientas avanzadas de análisis y predicción de ventas 
                adaptadas específicamente a las necesidades de pequeños supermercados, permitiéndoles tomar decisiones 
                informadas basadas en datos.
                """
            },
            {
                "title": "¿Qué Puede Hacer Esta Herramienta Por Su Negocio?",
                "content": """
                Nuestra herramienta de análisis y predicción de ventas le permite:
                
                - **Predecir ventas futuras** con alta precisión para mejorar la planificación de inventario
                - **Identificar factores clave** que influyen en sus ventas
                - **Analizar patrones** por categoría de producto, sucursal, día de la semana y hora del día
                - **Comparar diferentes escenarios** para optimizar decisiones de negocio
                - **Visualizar tendencias** de manera clara e intuitiva
                - **Exportar resultados** para su uso en informes y presentaciones
                """
            },
            {
                "title": "Hallazgos Clave para Supermercados",
                "content": """
                Nuestro análisis de datos de ventas de supermercado ha revelado varios hallazgos importantes:
                
                ### Patrones Temporales
                
                - Los [días de la semana] muestran las mayores ventas, con picos entre [horas]
                - Existe una estacionalidad mensual con mayores ventas en [meses]
                
                ### Categorías de Productos
                
                - Las categorías más rentables son [categorías]
                - Las categorías con mayor volumen de ventas son [categorías]
                
                ### Comportamiento del Cliente
                
                - Los clientes [tipo de cliente] generan [porcentaje]% más ingresos
                - La satisfacción del cliente (calificación) es mayor para [categorías/condiciones]
                
                ### Métodos de Pago
                
                - Los pagos con [método] están asociados con tickets promedio más altos
                - Los pagos con [método] son más frecuentes en [condiciones]
                """
            },
            {
                "title": "Recomendaciones Prácticas",
                "content": """
                Basado en nuestro análisis, recomendamos las siguientes estrategias para optimizar sus operaciones:
                
                ### Gestión de Inventario
                
                - Aumente el stock de [productos] durante [períodos]
                - Reduzca el inventario de [productos] durante [períodos]
                - Implemente un sistema de reposición basado en predicciones para [categorías]
                
                ### Promociones y Marketing
                
                - Dirija promociones de [productos] a [segmento de clientes]
                - Programe ofertas especiales durante [períodos de baja demanda]
                - Cree paquetes de productos combinando [categorías complementarias]
                
                ### Operaciones
                
                - Ajuste los niveles de personal para [horas pico]
                - Optimice la distribución de productos en tienda basándose en [patrones]
                - Considere extender/ajustar horarios de operación durante [períodos]
                
                ### Experiencia del Cliente
                
                - Mejore la experiencia en [áreas con baja calificación]
                - Implemente programas de fidelización enfocados en [segmentos de clientes]
                - Capacite al personal en [áreas de oportunidad]
                """
            },
            {
                "title": "Cómo Usar la Herramienta",
                "content": """
                Nuestra herramienta está diseñada para ser intuitiva y fácil de usar, incluso sin conocimientos técnicos avanzados:
                
                1. **Exploración de Datos**: Analice sus datos históricos con visualizaciones interactivas
                2. **Predicción de Ventas**: Utilice diferentes modelos para predecir ventas futuras
                3. **Comparación de Escenarios**: Evalúe diferentes estrategias y su impacto potencial
                4. **Exportación de Resultados**: Guarde análisis y predicciones para uso posterior
                
                La herramienta incluye tutoriales integrados y ejemplos para ayudarle a aprovechar al máximo sus capacidades.
                """
            },
            {
                "title": "Casos de Éxito",
                "content": """
                Varios pequeños supermercados han implementado nuestras recomendaciones con resultados notables:
                
                - **Supermercado A**: Aumentó sus ventas en un 15% y redujo el desperdicio en un 20%
                - **Supermercado B**: Mejoró la satisfacción del cliente de 6.8 a 8.5 en una escala de 10
                - **Supermercado C**: Optimizó su inventario, liberando un 12% de capital de trabajo
                
                Estos resultados demuestran el potencial de la toma de decisiones basada en datos para transformar 
                la operación de pequeños supermercados.
                """
            },
            {
                "title": "Próximos Pasos",
                "content": """
                Para comenzar a optimizar su negocio con nuestra herramienta:
                
                1. **Descargue la aplicación** desde [enlace]
                2. **Importe sus datos** siguiendo la guía de inicio rápido
                3. **Explore el análisis automático** generado para su negocio
                4. **Experimente con predicciones** para diferentes escenarios
                5. **Implemente las recomendaciones** más relevantes para su contexto
                
                También ofrecemos servicios de consultoría personalizada para ayudarle a implementar 
                estas soluciones en su negocio específico.
                """
            },
            {
                "title": "Recursos Adicionales",
                "content": """
                - **Guía de Usuario Completa**: [enlace]
                - **Videos Tutoriales**: [enlace]
                - **Preguntas Frecuentes**: [enlace]
                - **Soporte Técnico**: [contacto]
                - **Blog con Actualizaciones**: [enlace]
                """
            }
        ]
    }
    
    return structure

def save_model_description_to_file(model_name, output_file):
    """
    Guarda la descripción de un modelo en un archivo.
    
    Args:
        model_name (str): Nombre del modelo.
        output_file (str): Ruta del archivo de salida.
    
    Returns:
        bool: True si se guardó correctamente, False en caso contrario.
    """
    try:
        description = generate_model_description(model_name)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(description)
        
        return True
    except Exception as e:
        print(f"Error al guardar la descripción del modelo: {e}")
        return False

def save_academic_report_structure_to_file(output_file):
    """
    Guarda la estructura del informe académico en un archivo.
    
    Args:
        output_file (str): Ruta del archivo de salida.
    
    Returns:
        bool: True si se guardó correctamente, False en caso contrario.
    """
    try:
        structure = generate_academic_report_structure()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"# {structure['title']}\n\n")
            
            for section in structure['sections']:
                f.write(f"## {section['title']}\n\n")
                f.write(f"{section['content']}\n\n")
        
        return True
    except Exception as e:
        print(f"Error al guardar la estructura del informe académico: {e}")
        return False

def save_web_publication_structure_to_file(output_file):
    """
    Guarda la estructura de la publicación web en un archivo.
    
    Args:
        output_file (str): Ruta del archivo de salida.
    
    Returns:
        bool: True si se guardó correctamente, False en caso contrario.
    """
    try:
        structure = generate_web_publication_structure()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"# {structure['title']}\n\n")
            
            for section in structure['sections']:
                f.write(f"## {section['title']}\n\n")
                f.write(f"{section['content']}\n\n")
        
        return True
    except Exception as e:
        print(f"Error al guardar la estructura de la publicación web: {e}")
        return False
