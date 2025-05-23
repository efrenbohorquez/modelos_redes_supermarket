"""
Script para verificar el manejo de diferentes formatos de hora
"""
import pandas as pd
import numpy as np

def test_time_format_handling():
    print("Probando manejo de diferentes formatos de hora...")
    
    # Lista de posibles formatos de hora que podrían estar en los datos
    time_formats = [
        ':00',         # Formato problemático que causa el error
        '10:30',       # Formato HH:MM estándar
        '14:45:00',    # Formato HH:MM:SS
        '9:05 AM',     # Formato con AM/PM
        '18:30:45',    # Formato con segundos
        '2022-05-15 14:30:00'  # Formato datetime completo
    ]
    
    print("\nPrueba con pd.to_datetime y format='%H:%M':")
    for time_str in time_formats:
        try:
            result = pd.to_datetime(time_str, format='%H:%M')
            print(f"  - '{time_str}' -> {result.hour}")
        except Exception as e:
            print(f"  - '{time_str}' -> Error: {e}")
    
    print("\nPrueba con pd.to_datetime y format='mixed':")
    for time_str in time_formats:
        try:
            result = pd.to_datetime(time_str, format='mixed', errors='coerce')
            if pd.notna(result):
                print(f"  - '{time_str}' -> {result.hour}")
            else:
                print(f"  - '{time_str}' -> Resultado es NaT")
        except Exception as e:
            print(f"  - '{time_str}' -> Error: {e}")
    
    print("\nPrueba con extracción de regex:")
    for time_str in time_formats:
        try:
            result = pd.Series([time_str]).astype(str).str.extract(r'(\d+)')[0].astype(float).iloc[0]
            print(f"  - '{time_str}' -> {int(result)}")
        except Exception as e:
            print(f"  - '{time_str}' -> Error: {e}")
    
    print("\nSolución recomendada - Enfoque combinado:")
    for time_str in time_formats:
        try:
            # Primero intentar con format='mixed'
            result = pd.to_datetime(time_str, format='mixed', errors='coerce')
            
            if pd.notna(result):
                print(f"  - '{time_str}' -> {result.hour} (usando to_datetime)")
            else:
                # Si falla, intentar extraer con regex
                result = pd.Series([time_str]).astype(str).str.extract(r'(\d+)')[0].astype(float).iloc[0]
                print(f"  - '{time_str}' -> {int(result)} (usando regex)")
        except Exception as e:
            print(f"  - '{time_str}' -> Error: {e}")

if __name__ == "__main__":
    test_time_format_handling()
