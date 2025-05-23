Write-Host "Deteniendo contenedores existentes..."
docker stop $(docker ps -q --filter ancestor=supermarket-sales-prediction) 2>$null
docker rm $(docker ps -a -q --filter ancestor=supermarket-sales-prediction) 2>$null

Write-Host "`nReconstruyendo imagen Docker con el código corregido..."
docker build -t supermarket-sales-prediction:corrected .

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✅ ¡Imagen Docker construida correctamente!"
    
    Write-Host "`nEjecutando el contenedor..."
    docker run -d -p 8501:8501 --name supermarket-prediction-corrected supermarket-sales-prediction:corrected
    
    Write-Host "`n✅ ¡Contenedor Docker ejecutado correctamente!"
    Write-Host "La aplicación estará disponible en http://localhost:8501"
} else {
    Write-Host "`n❌ Error al construir la imagen Docker. Revisa los errores anteriores."
}
