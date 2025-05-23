"""
Script para corregir problemas de formato de hora en app.py
"""
import re

# Leer el archivo app.py
with open('app.py', 'r', encoding='utf-8') as file:
    content = file.read()

# Patrón a buscar y reemplazar
pattern = r"data\['Hour'\] = pd\.to_datetime\(data\['Time'\], format='%H:%M'\)\.dt\.hour"
replacement = """try:
                # Primero intentamos con el formato específico
                data['Hour'] = pd.to_datetime(data['Time'], format='%H:%M').dt.hour
            except ValueError:
                # Si falla, intentamos con formato mixto
                data['Hour'] = pd.to_datetime(data['Time'], format='mixed').dt.hour
            except Exception as e:
                # Si todo falla, extraemos la hora manualmente
                # Extraer la hora de las cadenas de texto usando expresiones regulares
                data['Hour'] = data['Time'].astype(str).str.extract(r'(\\d+)').astype(float)"""

# Reemplazar todas las ocurrencias
modified_content = re.sub(pattern, replacement, content)

# Guardar el archivo modificado
with open('app.py', 'w', encoding='utf-8') as file:
    file.write(modified_content)

print("Archivo app.py actualizado correctamente.")
