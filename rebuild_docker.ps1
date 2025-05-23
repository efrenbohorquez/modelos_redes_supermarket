# Script para recrear Dockerfile y construir imagen

# Recrear el Dockerfile
Write-Host "Recreando el Dockerfile..." -ForegroundColor Yellow
Get-Content -Path "dockerfile_content.txt" | Set-Content -Path "Dockerfile.txt"
Move-Item -Path "Dockerfile.txt" -Destination "Dockerfile" -Force

# Verificar que el Dockerfile se ha creado correctamente
$dockerfileContent = Get-Content -Path "Dockerfile"
if ($dockerfileContent) {
    Write-Host "Dockerfile recreado exitosamente con $($dockerfileContent.Count) líneas." -ForegroundColor Green
} else {
    Write-Host "Error al recrear Dockerfile." -ForegroundColor Red
    exit 1
}

# Verificando si hay contenedores previos
Write-Host "Verificando si hay contenedores previos..." -ForegroundColor Yellow
$containerId = docker ps -a -q --filter "name=supermarket-sales"
if ($containerId) {
    Write-Host "Eliminando contenedor existente..." -ForegroundColor Yellow
    docker stop $containerId
    docker rm $containerId
}

# Construir la imagen
Write-Host "Construyendo la imagen Docker..." -ForegroundColor Green
docker build -t supermarket-sales-prediction .

# Verificar si la construcción fue exitosa
if ($LASTEXITCODE -eq 0) {
    Write-Host "Imagen construida exitosamente. Iniciando contenedor..." -ForegroundColor Green
    docker run -d --name supermarket-sales -p 8501:8501 `
        -v "${PWD}/models:/app/models" `
        -v "${PWD}/data:/app/data" `
        supermarket-sales-prediction
    
    # Verificar si se inició correctamente
    $containerId = docker ps -q --filter "name=supermarket-sales"
    if ($containerId) {
        Write-Host "¡Contenedor iniciado con éxito!" -ForegroundColor Green
        Write-Host "La aplicación está disponible en: http://localhost:8501" -ForegroundColor Cyan
        Write-Host "Mostrando los logs del contenedor:" -ForegroundColor Yellow
        Start-Sleep -Seconds 2
        docker logs supermarket-sales
    } else {
        Write-Host "Error al iniciar el contenedor. Verificando logs..." -ForegroundColor Red
        docker logs supermarket-sales
    }
} else {
    Write-Host "Error al construir la imagen Docker." -ForegroundColor Red
}
