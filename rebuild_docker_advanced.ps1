# Script para reconstruir y ejecutar el contenedor Docker con correcciones

# Detener y eliminar el contenedor existente si está ejecutándose
Write-Host "Deteniendo contenedores existentes..."
docker stop $(docker ps -q --filter ancestor=supermarket-sales-prediction) 2>$null
docker rm $(docker ps -a -q --filter ancestor=supermarket-sales-prediction) 2>$null

# Aplicar correcciones al código antes de construir
Write-Host "Aplicando correcciones al formato de hora..."
python fix_time_format_advanced.py

# Reconstruir la imagen con las correcciones
Write-Host "Reconstruyendo la imagen Docker..."
docker build -t supermarket-sales-prediction:fixed .

# Ejecutar el nuevo contenedor
Write-Host "Ejecutando el nuevo contenedor..."
docker run -d -p 8501:8501 --name supermarket-prediction-fixed supermarket-sales-prediction:fixed

Write-Host ""
Write-Host "¡El contenedor Docker se ha reconstruido y ejecutado con las correcciones de formato de hora!"
Write-Host "Puedes acceder a la aplicación en http://localhost:8501"
