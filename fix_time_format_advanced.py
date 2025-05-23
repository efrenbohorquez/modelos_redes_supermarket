"""
Script para corregir problemas de formato de hora en app.py
"""
import re

def fix_time_format_issues():
    print("Corrigiendo problemas de formato de hora en app.py...")
    
    with open('app.py', 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Definir los patrones de reemplazo
    replacements = [
        # Patrón 1: Formateo básico de hora
        (
            r"data\['Hour'\] = pd\.to_datetime\(data\['Time'\], format='%H:%M'\)\.dt\.hour",
            """data['Hour'] = None  # Inicializar columna
try:
    # Intentar formato flexible que funciona con diversas entradas
    data['Hour'] = pd.to_datetime(data['Time'], format='mixed').dt.hour
except Exception as e:
    # Si todo falla, extraer la hora manualmente
    print(f"Error al procesar horas: {e}")
    data['Hour'] = data['Time'].astype(str).str.extract(r'(\\d+)').astype(float)"""
        ),
        
        # Patrón 2: En visualizaciones avanzadas
        (
            r"if 'Hour' not in data\.columns:\s*\n\s*data\['Hour'\] = pd\.to_datetime\(data\['Time'\], format='%H:%M'\)\.dt\.hour",
            """if 'Hour' not in data.columns:
            try:
                # Intentar formato flexible
                data['Hour'] = pd.to_datetime(data['Time'], format='mixed').dt.hour
            except Exception as e:
                # Extraer manualmente
                print(f"Error al procesar horas: {e}")
                data['Hour'] = data['Time'].astype(str).str.extract(r'(\\d+)').astype(float)"""
        ),
        
        # Patrón 3: Fix para el bug en la sección ya corregida
        (
            r"elif viz_type == \"Patrones por Hora del Día\":\s*\n\s*# Ventas por hora del día\s*\n\s*try:\s*\n\s*# Intentar diferentes formatos de hora para manejar todas las posibles variaciones\s*\n\s*try:\s*\n\s*# Primero intentamos con el formato específico\s*\n\s*data\['Hour'\] = pd\.to_datetime\(data\['Time'\], format='%H:%M'\)\.dt\.hour\s*\n\s*except ValueError:\s*\n\s*# Si falla, intentamos con formato mixto\s*\n\s*data\['Hour'\] = pd\.to_datetime\(data\['Time'\], format='mixed'\)\.dt\.hour\s*\n\s*except Exception as e:\s*\n\s*# Si todo falla, extraemos la hora manualmente\s*\n\s*st\.warning\(f\"Se encontraron problemas al procesar los formatos de hora: {e}\"\)\s*\n\s*# Extraer la hora de las cadenas de texto usando expresiones regulares\s*\n\s*data\['Hour'\] = data\['Time'\]\.astype\(str\)\.str\.extract\(r'\(\\d\+\)'\)\.astype\(float\)",
            """elif viz_type == "Patrones por Hora del Día":
        # Ventas por hora del día
        data['Hour'] = None  # Inicializar columna
        try:
            # Intentar formato flexible
            data['Hour'] = pd.to_datetime(data['Time'], format='mixed').dt.hour
        except Exception as e:
            # Si falla, extraer manualmente
            st.warning(f"Se encontraron problemas al procesar los formatos de hora: {e}")
            data['Hour'] = data['Time'].astype(str).str.extract(r'(\\d+)').astype(float)"""
        )
    ]
    
    # Aplicar todos los reemplazos
    modified_content = content
    for pattern, replacement in replacements:
        modified_content = re.sub(pattern, replacement, modified_content)
    
    # Guardar el archivo modificado
    with open('app.py', 'w', encoding='utf-8') as file:
        file.write(modified_content)
    
    print("Archivo app.py actualizado correctamente.")

if __name__ == "__main__":
    fix_time_format_issues()
