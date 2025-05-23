import pandas as pd

print("Probando manejo de formato problemático ':00':")

# Probar con format='%H:%M'
try:
    result = pd.to_datetime(':00', format='%H:%M')
    print(f"Usando format='%H:%M' -> {result.hour}")
except Exception as e:
    print(f"Usando format='%H:%M' -> Error: {e}")

# Probar con format='mixed'
try:
    result = pd.to_datetime(':00', format='mixed', errors='coerce')
    if pd.notna(result):
        print(f"Usando format='mixed' -> {result.hour}")
    else:
        print("Usando format='mixed' -> NaT (no se pudo convertir)")
except Exception as e:
    print(f"Usando format='mixed' -> Error: {e}")

# Probar con regex
try:
    result = pd.Series([':00']).astype(str).str.extract(r'(\d+)')[0]
    print(f"Usando regex -> '{result.values[0] if not result.empty else 'No match'}'")
except Exception as e:
    print(f"Usando regex -> Error: {e}")

# Enfoque combinado (solución recomendada)
try:
    time_str = ':00'
    result = pd.to_datetime(time_str, format='mixed', errors='coerce')
    
    if pd.notna(result):
        print(f"Solución combinada -> {result.hour} (usando to_datetime)")
    else:
        # Si falla, intentar extraer con regex
        result = pd.Series([time_str]).astype(str).str.extract(r'(\d+)')[0]
        value = result.iloc[0] if not result.empty else None
        if value is not None:
            print(f"Solución combinada -> {value} (usando regex)")
        else:
            print("Solución combinada -> No se pudo extraer la hora")
except Exception as e:
    print(f"Solución combinada -> Error: {e}")
