# Construir y ejecutar la aplicación de predicción de ventas de supermercado
# Este script automatiza el proceso de construcción y ejecución del contenedor Docker

# Verificar si Docker está disponible
try {
    $dockerVersion = docker --version
    Write-Host "Docker encontrado: $dockerVersion" -ForegroundColor Green
}
catch {
    Write-Host "❌ Error: Docker no está instalado o no está disponible en el PATH." -ForegroundColor Red
    Write-Host "Por favor, instala Docker y asegúrate de que está en tu PATH." -ForegroundColor Red
    exit 1
}

# Verificar si los directorios necesarios existen
if (-not (Test-Path -Path "./models")) {
    Write-Host "❌ Error: El directorio 'models' no existe en la ubicación actual." -ForegroundColor Red
    exit 1
}

if (-not (Test-Path -Path "./data")) {
    Write-Host "❌ Error: El directorio 'data' no existe en la ubicación actual." -ForegroundColor Red
    exit 1
}

Write-Host "Construyendo la imagen de Docker..." -ForegroundColor Green
docker build -t supermarket-sales-prediction . 2>&1

# Verificar si la construcción fue exitosa
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Error: No se pudo construir la imagen Docker." -ForegroundColor Red
    exit 1
}

Write-Host "✅ Imagen Docker construida exitosamente." -ForegroundColor Green

Write-Host "Verificando si el contenedor ya está en ejecución..." -ForegroundColor Yellow
$containerId = docker ps -q --filter "name=supermarket-sales"

if ($containerId) {
    Write-Host "Deteniendo el contenedor existente..." -ForegroundColor Yellow
    docker stop $containerId
    docker rm $containerId
}

Write-Host "Iniciando el contenedor de la aplicación..." -ForegroundColor Green
docker run -d --name supermarket-sales -p 8501:8501 `
    -v "${PWD}/models:/app/models" `
    -v "${PWD}/data:/app/data" `
    supermarket-sales-prediction

# Verificar si el contenedor se inició correctamente
$containerId = docker ps -q --filter "name=supermarket-sales"
if (-not $containerId) {
    Write-Host "❌ Error: No se pudo iniciar el contenedor." -ForegroundColor Red
    Write-Host "Revisando los logs:" -ForegroundColor Yellow
    docker logs supermarket-sales 2>&1
    exit 1
}

Write-Host "✅ Contenedor iniciado correctamente." -ForegroundColor Green
Write-Host ""
Write-Host "La aplicación está disponible en: http://localhost:8501" -ForegroundColor Cyan
Write-Host ""
Write-Host "Para detener el contenedor, ejecuta: docker stop supermarket-sales" -ForegroundColor Yellow

Write-Host "¡La aplicación está en ejecución!" -ForegroundColor Cyan
Write-Host "Accede a la aplicación en: http://localhost:8501" -ForegroundColor Cyan
