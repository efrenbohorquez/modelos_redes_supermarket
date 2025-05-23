# Parche directo para el app.py
# Este script debe copiarse manualmente al contenedor Docker

import re

# Función para formatear la hora de manera segura
def safe_format_hour(time_str):
    """Convierte la cadena de tiempo al formato de hora de manera segura."""
    if isinstance(time_str, str):
        # Intenta extraer solo el número de la hora
        match = re.search(r'(\d+)', time_str)
        if match:
            return int(match.group(1))
    return 0  # valor por defecto si no se puede extraer

# Reemplazar todas las conversiones de hora en el código
def process_time_safely(df):
    """Procesa la columna Time de manera segura para extraer la hora."""
    if 'Time' in df.columns:
        df['Hour'] = df['Time'].apply(safe_format_hour)
    return df

# Este código se debe añadir al inicio del archivo app.py, 
# justo después de las importaciones y antes de cualquier
# procesamiento de datos.
