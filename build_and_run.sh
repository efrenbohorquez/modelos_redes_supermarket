#!/bin/bash
# Construir y ejecutar la aplicación de predicción de ventas de supermercado
# Este script automatiza el proceso de construcción y ejecución del contenedor Docker

# Verificar si Docker está disponible
if ! command -v docker &> /dev/null; then
    echo -e "\e[31m❌ Error: Docker no está instalado o no está disponible en el PATH.\e[0m"
    echo -e "\e[31mPor favor, instala Docker y asegúrate de que está en tu PATH.\e[0m"
    exit 1
fi

echo -e "\e[32mDocker encontrado: $(docker --version)\e[0m"

# Verificar si los directorios necesarios existen
if [ ! -d "./models" ]; then
    echo -e "\e[31m❌ Error: El directorio 'models' no existe en la ubicación actual.\e[0m"
    exit 1
fi

if [ ! -d "./data" ]; then
    echo -e "\e[31m❌ Error: El directorio 'data' no existe en la ubicación actual.\e[0m"
    exit 1
fi

echo -e "\e[32mConstruyendo la imagen de Docker...\e[0m"
if ! docker build -t supermarket-sales-prediction .; then
    echo -e "\e[31m❌ Error: No se pudo construir la imagen Docker.\e[0m"
    exit 1
fi

echo -e "\e[32m✅ Imagen Docker construida exitosamente.\e[0m"

echo -e "\e[33mVerificando si el contenedor ya está en ejecución...\e[0m"
containerId=$(docker ps -q --filter "name=supermarket-sales")

if [ ! -z "$containerId" ]; then
    echo -e "\e[33mDeteniendo el contenedor existente...\e[0m"
    docker stop $containerId
    docker rm $containerId
fi

echo -e "\e[32mIniciando el contenedor de la aplicación...\e[0m"
docker run -d --name supermarket-sales -p 8501:8501 \
    -v "$(pwd)/models:/app/models" \
    -v "$(pwd)/data:/app/data" \
    supermarket-sales-prediction

# Verificar si el contenedor se inició correctamente
containerId=$(docker ps -q --filter "name=supermarket-sales")
if [ -z "$containerId" ]; then
    echo -e "\e[31m❌ Error: No se pudo iniciar el contenedor.\e[0m"
    echo -e "\e[33mRevisando los logs:\e[0m"
    docker logs supermarket-sales
    exit 1
fi

echo -e "\e[32m✅ Contenedor iniciado correctamente.\e[0m"
echo
echo -e "\e[36mLa aplicación está disponible en: http://localhost:8501\e[0m"
echo
echo -e "\e[33mPara detener el contenedor, ejecuta: docker stop supermarket-sales\e[0m"

echo -e "\e[36m¡La aplicación está en ejecución!\e[0m"
echo -e "\e[36mAccede a la aplicación en: http://localhost:8501\e[0m"
