# Construir y ejecutar la imagen de Docker manualmente

# Eliminar contenedores antiguos si existen
Write-Host "Verificando y eliminando contenedores existentes..." -ForegroundColor Yellow
$containerId = docker ps -a -q --filter "name=supermarket-sales"
if ($containerId) {
    docker stop $containerId
    docker rm $containerId
    Write-Host "Contenedor existente eliminado." -ForegroundColor Green
}

# Construir la imagen
Write-Host "Construyendo la imagen Docker..." -ForegroundColor Green
docker build -t supermarket-sales-prediction .

# Verificar si la construcción fue exitosa
if ($LASTEXITCODE -eq 0) {
    Write-Host "Iniciando el contenedor..." -ForegroundColor Green
    docker run -d --name supermarket-sales -p 8501:8501 `
        -v "${PWD}/models:/app/models" `
        -v "${PWD}/data:/app/data" `
        supermarket-sales-prediction
    
    # Verificar si se inició correctamente
    $running = docker ps -q --filter "name=supermarket-sales"
    if ($running) {
        Write-Host "¡Contenedor iniciado con éxito!" -ForegroundColor Green
        Write-Host "La aplicación está disponible en: http://localhost:8501" -ForegroundColor Cyan
    } else {
        Write-Host "Error al iniciar el contenedor. Verificando logs..." -ForegroundColor Red
        docker logs supermarket-sales
    }
} else {
    Write-Host "Error al construir la imagen Docker." -ForegroundColor Red
}
