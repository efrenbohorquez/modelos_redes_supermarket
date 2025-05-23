# Script para reconstruir el contenedor Docker con la versión final del Dockerfile

# Detener y eliminar contenedores existentes
Write-Host "Deteniendo contenedores existentes..."
docker stop $(docker ps -q --filter name=supermarket) 2>$null
docker rm $(docker ps -a -q --filter name=supermarket) 2>$null

# Construir la nueva imagen usando Dockerfile.final
Write-Host "Construyendo nueva imagen con Dockerfile.final..."
docker build -t supermarket-sales-prediction:final -f Dockerfile.final .

# Verificar si la construcción fue exitosa
if ($LASTEXITCODE -eq 0) {
    # Ejecutar el nuevo contenedor
    Write-Host "Ejecutando el nuevo contenedor..."
    docker run -d -p 8501:8501 --name supermarket-prediction-final supermarket-sales-prediction:final
    
    Write-Host ""
    Write-Host "¡El contenedor Docker se ha reconstruido y ejecutado con las correcciones!"
    Write-Host "La aplicación estará disponible en http://localhost:8501"
    Write-Host ""
    Write-Host "Para ver los logs del contenedor, ejecuta:"
    Write-Host "docker logs supermarket-prediction-final"
} else {
    Write-Host "Error al construir la imagen Docker. Revisa los errores anteriores."
}
