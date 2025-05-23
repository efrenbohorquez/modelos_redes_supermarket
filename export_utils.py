"""
Utilidades para exportación de resultados y comparación de modelos.

Este módulo proporciona funciones para exportar resultados de análisis,
predicciones y comparaciones de modelos en diferentes formatos,
facilitando la documentación y presentación de resultados.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64
import json
import os
import pickle
import joblib
from datetime import datetime
import xlsxwriter
from fpdf2 import FPDF
import csv
import webbrowser

def export_dataframe_to_excel(df, filename, sheet_name='Data', include_index=False):
    """
    Exporta un DataFrame a un archivo Excel.
    
    Args:
        df (pd.DataFrame): DataFrame a exportar.
        filename (str): Nombre del archivo de salida.
        sheet_name (str): Nombre de la hoja de cálculo.
        include_index (bool): Si se debe incluir el índice.
    
    Returns:
        str: Ruta completa del archivo guardado.
    """
    # Asegurar que la extensión sea .xlsx
    if not filename.endswith('.xlsx'):
        filename += '.xlsx'
    
    # Crear directorio si no existe
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    
    # Exportar a Excel
    df.to_excel(filename, sheet_name=sheet_name, index=include_index)
    
    return os.path.abspath(filename)

def export_dataframe_to_csv(df, filename, include_index=False):
    """
    Exporta un DataFrame a un archivo CSV.
    
    Args:
        df (pd.DataFrame): DataFrame a exportar.
        filename (str): Nombre del archivo de salida.
        include_index (bool): Si se debe incluir el índice.
    
    Returns:
        str: Ruta completa del archivo guardado.
    """
    # Asegurar que la extensión sea .csv
    if not filename.endswith('.csv'):
        filename += '.csv'
    
    # Crear directorio si no existe
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    
    # Exportar a CSV
    df.to_csv(filename, index=include_index)
    
    return os.path.abspath(filename)

def export_dataframe_to_json(df, filename, orient='records'):
    """
    Exporta un DataFrame a un archivo JSON.
    
    Args:
        df (pd.DataFrame): DataFrame a exportar.
        filename (str): Nombre del archivo de salida.
        orient (str): Orientación del JSON ('records', 'split', 'index', 'columns', 'values').
    
    Returns:
        str: Ruta completa del archivo guardado.
    """
    # Asegurar que la extensión sea .json
    if not filename.endswith('.json'):
        filename += '.json'
    
    # Crear directorio si no existe
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    
    # Exportar a JSON
    df.to_json(filename, orient=orient)
    
    return os.path.abspath(filename)

def export_figure_to_image(fig, filename, format='png', dpi=300):
    """
    Exporta una figura de matplotlib a un archivo de imagen.
    
    Args:
        fig: Figura de matplotlib.
        filename (str): Nombre del archivo de salida.
        format (str): Formato de la imagen ('png', 'jpg', 'svg', 'pdf').
        dpi (int): Resolución de la imagen.
    
    Returns:
        str: Ruta completa del archivo guardado.
    """
    # Asegurar que la extensión sea correcta
    if not filename.endswith(f'.{format}'):
        filename += f'.{format}'
    
    # Crear directorio si no existe
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    
    # Exportar figura
    fig.savefig(filename, format=format, dpi=dpi, bbox_inches='tight')
    
    return os.path.abspath(filename)

def export_plotly_figure_to_html(fig, filename):
    """
    Exporta una figura de Plotly a un archivo HTML interactivo.
    
    Args:
        fig: Figura de Plotly.
        filename (str): Nombre del archivo de salida.
    
    Returns:
        str: Ruta completa del archivo guardado.
    """
    # Asegurar que la extensión sea .html
    if not filename.endswith('.html'):
        filename += '.html'
    
    # Crear directorio si no existe
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    
    # Exportar figura
    fig.write_html(filename)
    
    return os.path.abspath(filename)

def export_plotly_figure_to_image(fig, filename, format='png', width=1200, height=800):
    """
    Exporta una figura de Plotly a un archivo de imagen estática.
    
    Args:
        fig: Figura de Plotly.
        filename (str): Nombre del archivo de salida.
        format (str): Formato de la imagen ('png', 'jpg', 'svg', 'pdf').
        width (int): Ancho de la imagen en píxeles.
        height (int): Alto de la imagen en píxeles.
    
    Returns:
        str: Ruta completa del archivo guardado.
    """
    # Asegurar que la extensión sea correcta
    if not filename.endswith(f'.{format}'):
        filename += f'.{format}'
    
    # Crear directorio si no existe
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    
    # Exportar figura
    fig.write_image(filename, format=format, width=width, height=height)
    
    return os.path.abspath(filename)

def export_model_results_to_excel(results, filename, include_metrics=True, include_predictions=True):
    """
    Exporta resultados de modelos a un archivo Excel con múltiples hojas.
    
    Args:
        results (dict): Diccionario con resultados de modelos.
        filename (str): Nombre del archivo de salida.
        include_metrics (bool): Si se deben incluir métricas.
        include_predictions (bool): Si se deben incluir predicciones.
    
    Returns:
        str: Ruta completa del archivo guardado.
    """
    # Asegurar que la extensión sea .xlsx
    if not filename.endswith('.xlsx'):
        filename += '.xlsx'
    
    # Crear directorio si no existe
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    
    # Crear Excel writer
    writer = pd.ExcelWriter(filename, engine='xlsxwriter')
    
    # Exportar métricas si se solicita
    if include_metrics:
        metrics_data = []
        
        for model_name, model_results in results.items():
            if isinstance(model_results, dict) and 'metrics' in model_results:
                metrics = model_results['metrics']
                metrics['Modelo'] = model_name
                metrics_data.append(metrics)
        
        if metrics_data:
            metrics_df = pd.DataFrame(metrics_data)
            metrics_df.to_excel(writer, sheet_name='Métricas', index=False)
    
    # Exportar predicciones si se solicita
    if include_predictions:
        for model_name, model_results in results.items():
            if isinstance(model_results, dict) and 'predictions' in model_results:
                predictions = model_results['predictions']
                
                if isinstance(predictions, dict):
                    for pred_name, pred_data in predictions.items():
                        sheet_name = f"{model_name}_{pred_name}"[:31]  # Limitar longitud del nombre de hoja
                        
                        if isinstance(pred_data, pd.DataFrame):
                            pred_data.to_excel(writer, sheet_name=sheet_name, index=False)
                        elif isinstance(pred_data, dict):
                            pd.DataFrame(pred_data).to_excel(writer, sheet_name=sheet_name, index=False)
                elif isinstance(predictions, pd.DataFrame):
                    sheet_name = f"{model_name}_pred"[:31]
                    predictions.to_excel(writer, sheet_name=sheet_name, index=False)
    
    # Guardar archivo
    writer.close()
    
    return os.path.abspath(filename)

def export_model_comparison_to_excel(comparison_df, filename, include_chart=True):
    """
    Exporta una comparación de modelos a un archivo Excel con gráfico.
    
    Args:
        comparison_df (pd.DataFrame): DataFrame con comparación de modelos.
        filename (str): Nombre del archivo de salida.
        include_chart (bool): Si se debe incluir un gráfico.
    
    Returns:
        str: Ruta completa del archivo guardado.
    """
    # Asegurar que la extensión sea .xlsx
    if not filename.endswith('.xlsx'):
        filename += '.xlsx'
    
    # Crear directorio si no existe
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    
    # Crear Excel writer
    writer = pd.ExcelWriter(filename, engine='xlsxwriter')
    
    # Exportar datos
    comparison_df.to_excel(writer, sheet_name='Comparación', index=False)
    
    # Añadir gráfico si se solicita
    if include_chart and 'RMSE' in comparison_df.columns:
        workbook = writer.book
        worksheet = writer.sheets['Comparación']
        
        # Crear gráfico
        chart = workbook.add_chart({'type': 'column'})
        
        # Configurar datos del gráfico
        chart.add_series({
            'name': 'RMSE',
            'categories': ['Comparación', 1, 0, len(comparison_df), 0],
            'values': ['Comparación', 1, comparison_df.columns.get_loc('RMSE'), len(comparison_df), comparison_df.columns.get_loc('RMSE')],
        })
        
        # Configurar gráfico
        chart.set_title({'name': 'Comparación de RMSE entre Modelos'})
        chart.set_x_axis({'name': 'Modelo'})
        chart.set_y_axis({'name': 'RMSE'})
        
        # Insertar gráfico
        worksheet.insert_chart('H2', chart)
    
    # Guardar archivo
    writer.close()
    
    return os.path.abspath(filename)

def export_predictions_to_excel(y_true, y_pred, dates=None, filename='predicciones.xlsx'):
    """
    Exporta predicciones vs valores reales a un archivo Excel con gráfico.
    
    Args:
        y_true (array): Valores reales.
        y_pred (array): Valores predichos.
        dates (array, optional): Fechas correspondientes.
        filename (str): Nombre del archivo de salida.
    
    Returns:
        str: Ruta completa del archivo guardado.
    """
    # Asegurar que la extensión sea .xlsx
    if not filename.endswith('.xlsx'):
        filename += '.xlsx'
    
    # Crear directorio si no existe
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    
    # Crear DataFrame
    data = {'Valor Real': y_true, 'Predicción': y_pred}
    
    if dates is not None:
        data['Fecha'] = dates
    
    df = pd.DataFrame(data)
    
    # Crear Excel writer
    writer = pd.ExcelWriter(filename, engine='xlsxwriter')
    
    # Exportar datos
    df.to_excel(writer, sheet_name='Predicciones', index=False)
    
    # Añadir gráfico
    workbook = writer.book
    worksheet = writer.sheets['Predicciones']
    
    # Crear gráfico
    chart = workbook.add_chart({'type': 'scatter'})
    
    # Configurar datos del gráfico
    chart.add_series({
        'name': 'Predicciones vs Reales',
        'categories': ['Predicciones', 1, 0, len(df), 0],
        'values': ['Predicciones', 1, 1, len(df), 1],
        'marker': {'type': 'circle', 'size': 7},
    })
    
    # Añadir línea de referencia (y=x)
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    
    chart.add_series({
        'name': 'Referencia (y=x)',
        'categories': ['Predicciones', 0, 0, 0, 0],
        'values': ['Predicciones', 0, 0, 0, 0],
        'line': {'color': 'red', 'dash_type': 'dash'},
        'marker': {'type': 'none'},
        'data_labels': {'none': True},
        'points': [
            {'x': min_val, 'y': min_val},
            {'x': max_val, 'y': max_val}
        ]
    })
    
    # Configurar gráfico
    chart.set_title({'name': 'Predicciones vs Valores Reales'})
    chart.set_x_axis({'name': 'Valor Real'})
    chart.set_y_axis({'name': 'Predicción'})
    
    # Insertar gráfico
    worksheet.insert_chart('E2', chart)
    
    # Guardar archivo
    writer.close()
    
    return os.path.abspath(filename)

def export_report_to_pdf(title, sections, filename='reporte.pdf', author='Sistema de Análisis de Ventas'):
    """
    Exporta un informe estructurado a un archivo PDF.
    
    Args:
        title (str): Título del informe.
        sections (list): Lista de diccionarios con 'title' y 'content'.
        filename (str): Nombre del archivo de salida.
        author (str): Autor del informe.
    
    Returns:
        str: Ruta completa del archivo guardado.
    """
    # Asegurar que la extensión sea .pdf
    if not filename.endswith('.pdf'):
        filename += '.pdf'
    
    # Crear directorio si no existe
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    
    # Crear PDF
    pdf = FPDF()
    pdf.add_page()
    
    # Configurar fuente
    pdf.set_font('helvetica', 'B', 16)
    
    # Título
    pdf.cell(0, 10, title, 0, 1, 'C')
    pdf.ln(10)
    
    # Autor y fecha
    pdf.set_font('helvetica', 'I', 10)
    pdf.cell(0, 5, f"Autor: {author}", 0, 1, 'R')
    pdf.cell(0, 5, f"Fecha: {datetime.now().strftime('%d/%m/%Y')}", 0, 1, 'R')
    pdf.ln(10)
    
    # Secciones
    for section in sections:
        # Título de sección
        pdf.set_font('helvetica', 'B', 14)
        pdf.cell(0, 10, section['title'], 0, 1, 'L')
        pdf.ln(2)
        
        # Contenido de sección
        pdf.set_font('helvetica', '', 12)
        pdf.multi_cell(0, 8, section['content'])
        pdf.ln(5)
    
    # Guardar archivo
    pdf.output(filename)
    
    return os.path.abspath(filename)

def create_download_link(file_path, link_text=None):
    """
    Crea un enlace HTML para descargar un archivo.
    
    Args:
        file_path (str): Ruta del archivo.
        link_text (str, optional): Texto del enlace.
    
    Returns:
        str: Enlace HTML para descargar el archivo.
    """
    if link_text is None:
        link_text = f"Descargar {os.path.basename(file_path)}"
    
    with open(file_path, 'rb') as f:
        data = f.read()
    
    b64 = base64.b64encode(data).decode()
    
    # Determinar tipo MIME
    ext = os.path.splitext(file_path)[1].lower()
    mime_types = {
        '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        '.csv': 'text/csv',
        '.json': 'application/json',
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.svg': 'image/svg+xml',
        '.pdf': 'application/pdf',
        '.html': 'text/html',
        '.txt': 'text/plain'
    }
    
    mime_type = mime_types.get(ext, 'application/octet-stream')
    
    href = f'<a href="data:{mime_type};base64,{b64}" download="{os.path.basename(file_path)}">{link_text}</a>'
    return href

def export_model_metrics_summary(models_metrics, filename='metricas_modelos.csv'):
    """
    Exporta un resumen de métricas de múltiples modelos a un archivo CSV.
    
    Args:
        models_metrics (dict): Diccionario con métricas de modelos.
        filename (str): Nombre del archivo de salida.
    
    Returns:
        str: Ruta completa del archivo guardado.
    """
    # Asegurar que la extensión sea .csv
    if not filename.endswith('.csv'):
        filename += '.csv'
    
    # Crear directorio si no existe
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    
    # Preparar datos
    data = []
    
    for model_name, metrics in models_metrics.items():
        row = {'Modelo': model_name}
        row.update(metrics)
        data.append(row)
    
    # Crear DataFrame
    df = pd.DataFrame(data)
    
    # Exportar a CSV
    df.to_csv(filename, index=False)
    
    return os.path.abspath(filename)

def export_feature_importance(feature_names, importance_values, filename='importancia_caracteristicas.csv'):
    """
    Exporta la importancia de características a un archivo CSV.
    
    Args:
        feature_names (list): Nombres de las características.
        importance_values (list): Valores de importancia.
        filename (str): Nombre del archivo de salida.
    
    Returns:
        str: Ruta completa del archivo guardado.
    """
    # Asegurar que la extensión sea .csv
    if not filename.endswith('.csv'):
        filename += '.csv'
    
    # Crear directorio si no existe
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    
    # Crear DataFrame
    df = pd.DataFrame({
        'Característica': feature_names,
        'Importancia': importance_values
    })
    
    # Ordenar por importancia
    df = df.sort_values('Importancia', ascending=False)
    
    # Exportar a CSV
    df.to_csv(filename, index=False)
    
    return os.path.abspath(filename)

def export_model_comparison_report(comparison_df, filename='comparacion_modelos.html'):
    """
    Exporta un informe de comparación de modelos a un archivo HTML.
    
    Args:
        comparison_df (pd.DataFrame): DataFrame con comparación de modelos.
        filename (str): Nombre del archivo de salida.
    
    Returns:
        str: Ruta completa del archivo guardado.
    """
    # Asegurar que la extensión sea .html
    if not filename.endswith('.html'):
        filename += '.html'
    
    # Crear directorio si no existe
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    
    # Crear figura de comparación
    fig = px.bar(comparison_df, x='Modelo', y='RMSE', 
                title='Comparación de RMSE entre Modelos',
                color='Modelo')
    
    # Crear HTML
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Comparación de Modelos</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                line-height: 1.6;
            }}
            h1 {{
                color: #2c3e50;
                border-bottom: 2px solid #3498db;
                padding-bottom: 10px;
            }}
            h2 {{
                color: #2980b9;
                margin-top: 30px;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 20px 0;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
            }}
            tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}
            .chart-container {{
                width: 100%;
                height: 500px;
                margin: 20px 0;
            }}
            .footer {{
                margin-top: 50px;
                font-size: 0.8em;
                color: #7f8c8d;
                text-align: center;
            }}
        </style>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    </head>
    <body>
        <h1>Comparación de Modelos de Predicción</h1>
        
        <h2>Tabla Comparativa</h2>
        {comparison_df.to_html(index=False)}
        
        <h2>Visualización Comparativa</h2>
        <div class="chart-container" id="chart"></div>
        
        <script>
            var plotlyData = {fig.to_json()};
            Plotly.newPlot('chart', plotlyData.data, plotlyData.layout);
        </script>
        
        <h2>Análisis de Resultados</h2>
        <p>
            La comparación de modelos muestra que el modelo {comparison_df.sort_values('RMSE').iloc[0]['Modelo']} 
            tiene el mejor rendimiento con un RMSE de {comparison_df.sort_values('RMSE').iloc[0]['RMSE']:.4f}.
        </p>
        <p>
            Los factores que pueden influir en el rendimiento de los modelos incluyen la complejidad del modelo,
            la cantidad de datos de entrenamiento, y la naturaleza de los patrones en los datos.
        </p>
        
        <div class="footer">
            Generado el {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
        </div>
    </body>
    </html>
    """
    
    # Guardar HTML
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return os.path.abspath(filename)

def export_sales_analysis_dashboard(data, filename='dashboard_ventas.html'):
    """
    Exporta un dashboard de análisis de ventas a un archivo HTML.
    
    Args:
        data (pd.DataFrame): DataFrame con datos de ventas.
        filename (str): Nombre del archivo de salida.
    
    Returns:
        str: Ruta completa del archivo guardado.
    """
    # Asegurar que la extensión sea .html
    if not filename.endswith('.html'):
        filename += '.html'
    
    # Crear directorio si no existe
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    
    # Preparar datos para visualizaciones
    # Ventas por categoría
    category_sales = data.groupby('Product line')['Total'].sum().reset_index()
    fig_category = px.bar(category_sales, x='Product line', y='Total', 
                         title='Ventas Totales por Categoría de Producto',
                         color='Product line')
    
    # Ventas por sucursal
    branch_sales = data.groupby('Branch')['Total'].sum().reset_index()
    fig_branch = px.bar(branch_sales, x='Branch', y='Total', 
                       title='Ventas Totales por Sucursal',
                       color='Branch')
    
    # Ventas por método de pago
    payment_sales = data.groupby('Payment')['Total'].sum().reset_index()
    fig_payment = px.pie(payment_sales, names='Payment', values='Total', 
                        title='Ventas por Método de Pago')
    
    # Ventas por género
    gender_sales = data.groupby('Gender')['Total'].sum().reset_index()
    fig_gender = px.pie(gender_sales, names='Gender', values='Total', 
                       title='Ventas por Género')
    
    # Crear HTML
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Dashboard de Análisis de Ventas</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                line-height: 1.6;
            }}
            h1 {{
                color: #2c3e50;
                border-bottom: 2px solid #3498db;
                padding-bottom: 10px;
                text-align: center;
            }}
            h2 {{
                color: #2980b9;
                margin-top: 30px;
            }}
            .dashboard-container {{
                display: flex;
                flex-wrap: wrap;
                justify-content: space-between;
            }}
            .chart-container {{
                width: 48%;
                height: 400px;
                margin-bottom: 30px;
            }}
            .metrics-container {{
                display: flex;
                justify-content: space-between;
                margin-bottom: 30px;
            }}
            .metric-card {{
                background-color: #f8f9fa;
                border-radius: 5px;
                padding: 15px;
                width: 23%;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                text-align: center;
            }}
            .metric-value {{
                font-size: 24px;
                font-weight: bold;
                color: #2980b9;
            }}
            .metric-label {{
                font-size: 14px;
                color: #7f8c8d;
            }}
            .footer {{
                margin-top: 50px;
                font-size: 0.8em;
                color: #7f8c8d;
                text-align: center;
            }}
        </style>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    </head>
    <body>
        <h1>Dashboard de Análisis de Ventas de Supermercado</h1>
        
        <div class="metrics-container">
            <div class="metric-card">
                <div class="metric-value">{data.shape[0]}</div>
                <div class="metric-label">Transacciones</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">${data['Total'].sum():,.2f}</div>
                <div class="metric-label">Ventas Totales</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">${data['Total'].mean():,.2f}</div>
                <div class="metric-label">Ticket Promedio</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{data['Rating'].mean():.2f}</div>
                <div class="metric-label">Calificación Promedio</div>
            </div>
        </div>
        
        <div class="dashboard-container">
            <div class="chart-container" id="chart1"></div>
            <div class="chart-container" id="chart2"></div>
            <div class="chart-container" id="chart3"></div>
            <div class="chart-container" id="chart4"></div>
        </div>
        
        <h2>Resumen de Hallazgos</h2>
        <p>
            El análisis de ventas muestra que la categoría de producto con mayores ventas es 
            <strong>{category_sales.sort_values('Total', ascending=False).iloc[0]['Product line']}</strong> 
            con un total de <strong>${category_sales.sort_values('Total', ascending=False).iloc[0]['Total']:,.2f}</strong>.
        </p>
        <p>
            La sucursal con mejor desempeño es 
            <strong>Sucursal {branch_sales.sort_values('Total', ascending=False).iloc[0]['Branch']}</strong> 
            con ventas totales de <strong>${branch_sales.sort_values('Total', ascending=False).iloc[0]['Total']:,.2f}</strong>.
        </p>
        <p>
            El método de pago más utilizado es 
            <strong>{payment_sales.sort_values('Total', ascending=False).iloc[0]['Payment']}</strong> 
            representando el <strong>{100 * payment_sales.sort_values('Total', ascending=False).iloc[0]['Total'] / payment_sales['Total'].sum():.1f}%</strong> 
            de las ventas totales.
        </p>
        
        <script>
            var plotlyData1 = {fig_category.to_json()};
            Plotly.newPlot('chart1', plotlyData1.data, plotlyData1.layout);
            
            var plotlyData2 = {fig_branch.to_json()};
            Plotly.newPlot('chart2', plotlyData2.data, plotlyData2.layout);
            
            var plotlyData3 = {fig_payment.to_json()};
            Plotly.newPlot('chart3', plotlyData3.data, plotlyData3.layout);
            
            var plotlyData4 = {fig_gender.to_json()};
            Plotly.newPlot('chart4', plotlyData4.data, plotlyData4.layout);
        </script>
        
        <div class="footer">
            Generado el {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
        </div>
    </body>
    </html>
    """
    
    # Guardar HTML
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return os.path.abspath(filename)

def export_model_to_file(model, filename):
    """
    Exporta un modelo entrenado a un archivo.
    
    Args:
        model: Modelo entrenado (scikit-learn, Keras, etc.).
        filename (str): Nombre del archivo de salida.
    
    Returns:
        str: Ruta completa del archivo guardado.
    """
    # Crear directorio si no existe
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    
    # Determinar tipo de modelo y formato adecuado
    if 'keras' in str(type(model)).lower():
        # Modelo de Keras
        if not filename.endswith('.h5'):
            filename += '.h5'
        model.save(filename)
    elif hasattr(model, 'predict'):
        # Modelo de scikit-learn u otro compatible con joblib
        if not filename.endswith('.joblib'):
            filename += '.joblib'
        joblib.dump(model, filename)
    else:
        # Otro tipo de objeto
        if not filename.endswith('.pkl'):
            filename += '.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
    
    return os.path.abspath(filename)

def export_all_results(data, models_results, output_dir='resultados'):
    """
    Exporta todos los resultados de análisis y modelos a una estructura de directorios.
    
    Args:
        data (pd.DataFrame): DataFrame con datos originales.
        models_results (dict): Diccionario con resultados de modelos.
        output_dir (str): Directorio de salida.
    
    Returns:
        dict: Diccionario con rutas de archivos exportados.
    """
    # Crear directorio principal si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Crear subdirectorios
    os.makedirs(os.path.join(output_dir, 'datos'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'modelos'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'visualizaciones'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'informes'), exist_ok=True)
    
    exported_files = {}
    
    # Exportar datos originales
    data_file = os.path.join(output_dir, 'datos', 'datos_originales.xlsx')
    exported_files['datos_originales'] = export_dataframe_to_excel(data, data_file)
    
    # Exportar resumen estadístico
    stats_file = os.path.join(output_dir, 'datos', 'resumen_estadistico.xlsx')
    exported_files['resumen_estadistico'] = export_dataframe_to_excel(data.describe(), stats_file)
    
    # Exportar resultados de modelos
    if models_results:
        # Comparación de modelos
        comparison_data = []
        
        for model_name, results in models_results.items():
            if isinstance(results, dict) and 'metrics' in results:
                metrics = results['metrics']
                metrics['Modelo'] = model_name
                comparison_data.append(metrics)
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            comparison_file = os.path.join(output_dir, 'modelos', 'comparacion_modelos.xlsx')
            exported_files['comparacion_modelos'] = export_model_comparison_to_excel(comparison_df, comparison_file)
            
            # Informe HTML de comparación
            html_file = os.path.join(output_dir, 'informes', 'comparacion_modelos.html')
            exported_files['informe_comparacion'] = export_model_comparison_report(comparison_df, html_file)
        
        # Resultados individuales de modelos
        for model_name, results in models_results.items():
            if isinstance(results, dict):
                model_dir = os.path.join(output_dir, 'modelos', model_name)
                os.makedirs(model_dir, exist_ok=True)
                
                # Exportar métricas
                if 'metrics' in results:
                    metrics_file = os.path.join(model_dir, 'metricas.json')
                    with open(metrics_file, 'w') as f:
                        json.dump(results['metrics'], f, indent=4)
                    exported_files[f'{model_name}_metricas'] = metrics_file
                
                # Exportar predicciones
                if 'predictions' in results and isinstance(results['predictions'], pd.DataFrame):
                    pred_file = os.path.join(model_dir, 'predicciones.xlsx')
                    exported_files[f'{model_name}_predicciones'] = export_dataframe_to_excel(results['predictions'], pred_file)
                
                # Exportar importancia de características
                if 'feature_importance' in results and isinstance(results['feature_importance'], dict):
                    imp_file = os.path.join(model_dir, 'importancia_caracteristicas.csv')
                    exported_files[f'{model_name}_importancia'] = export_feature_importance(
                        results['feature_importance']['features'],
                        results['feature_importance']['importance'],
                        imp_file
                    )
    
    # Exportar dashboard de ventas
    dashboard_file = os.path.join(output_dir, 'informes', 'dashboard_ventas.html')
    exported_files['dashboard_ventas'] = export_sales_analysis_dashboard(data, dashboard_file)
    
    # Crear archivo README con índice de archivos
    readme_content = f"""# Resultados de Análisis y Predicción de Ventas

## Contenido

### Datos
- [Datos Originales]({os.path.relpath(exported_files['datos_originales'], output_dir)})
- [Resumen Estadístico]({os.path.relpath(exported_files['resumen_estadistico'], output_dir)})

### Modelos
"""
    
    if 'comparacion_modelos' in exported_files:
        readme_content += f"- [Comparación de Modelos]({os.path.relpath(exported_files['comparacion_modelos'], output_dir)})\n"
    
    for key, path in exported_files.items():
        if key.endswith('_metricas'):
            model_name = key.replace('_metricas', '')
            readme_content += f"- {model_name}\n"
            readme_content += f"  - [Métricas]({os.path.relpath(path, output_dir)})\n"
            
            if f'{model_name}_predicciones' in exported_files:
                readme_content += f"  - [Predicciones]({os.path.relpath(exported_files[f'{model_name}_predicciones'], output_dir)})\n"
            
            if f'{model_name}_importancia' in exported_files:
                readme_content += f"  - [Importancia de Características]({os.path.relpath(exported_files[f'{model_name}_importancia'], output_dir)})\n"
    
    readme_content += """
### Informes
"""
    
    if 'informe_comparacion' in exported_files:
        readme_content += f"- [Informe de Comparación de Modelos]({os.path.relpath(exported_files['informe_comparacion'], output_dir)})\n"
    
    if 'dashboard_ventas' in exported_files:
        readme_content += f"- [Dashboard de Ventas]({os.path.relpath(exported_files['dashboard_ventas'], output_dir)})\n"
    
    readme_file = os.path.join(output_dir, 'README.md')
    with open(readme_file, 'w') as f:
        f.write(readme_content)
    
    exported_files['readme'] = readme_file
    
    return exported_files
