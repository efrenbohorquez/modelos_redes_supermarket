# Script para reconstruir y ejecutar el contenedor Docker con correcciones de hora

# Detener el contenedor existente si está ejecutándose
docker stop $(docker ps -q --filter name=supermarket-prediction) 2>/dev/null

# Aplicar correcciones al código antes de construir
python fix_time_format.py

# Reconstruir la imagen con las correcciones
docker build -t supermarket-sales-prediction:fixed .

# Ejecutar el contenedor con la nueva imagen
docker run -d -p 8501:8501 --name supermarket-prediction supermarket-sales-prediction:fixed

Write-Host "El contenedor Docker se ha reconstruido y ejecutado con las correcciones de formato de hora."
Write-Host "Puedes acceder a la aplicación en http://localhost:8501"
