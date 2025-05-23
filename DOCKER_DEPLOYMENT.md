# Implementación de Docker para el Proyecto de Predicción de Ventas

## Archivos y Cambios Implementados

Se han implementado los siguientes archivos para la containerización del proyecto:

### 1. Dockerfile
- Base: Python 3.11-slim
- Instalación de dependencias del sistema y Python
- Copia de archivos de proyecto
- Creación de estructura de directorios
- Verificación de modelos con script test_model_loading.py
- Configuración para ejecutar la aplicación Streamlit

### 2. docker-compose.yml
- Configuración para el servicio de la aplicación Streamlit
- Mapeo de puertos: 8501:8501
- Volúmenes para modelos y datos
- Variables de entorno para Streamlit

### 3. .dockerignore
- Exclusión de archivos y directorios innecesarios
- Optimización del contexto de construcción

### 4. Scripts de Automatización
- **build_and_run.ps1**: Script PowerShell para Windows
- **build_and_run.sh**: Script Bash para Linux/Mac
- **validate_docker_image.py**: Script para validar la construcción y funcionalidad de la imagen

## Problemas Conocidos y Soluciones

### Posibles Errores de TensorFlow
Si experimentas problemas al cargar modelos TensorFlow en el contenedor, pueden deberse a incompatibilidades entre las versiones de TensorFlow usadas para entrenar los modelos y las instaladas en el contenedor. Soluciones:

1. **Error de serialización de modelo**: Asegúrate de que la versión de TensorFlow en requirements_updated.txt sea compatible con tus modelos guardados.
2. **Errores de memoria**: TensorFlow puede consumir mucha memoria. Aumenta los recursos asignados a Docker.

### Problemas de Volumen
Si los modelos no se cargan correctamente:

1. Verifica que los volúmenes están montados correctamente
2. Comprueba los permisos de los directorios models/ y data/
3. Verifica que las rutas dentro del contenedor coinciden con las esperadas por la aplicación

### Errores de Red
Si no puedes acceder a la aplicación después de iniciar el contenedor:

1. Verifica que el puerto 8501 no esté siendo utilizado por otra aplicación
2. Comprueba si el firewall está bloqueando el puerto
3. Asegúrate de que Streamlit esté configurado para escuchar en 0.0.0.0 y no solo en localhost

## Comandos Útiles

### Construir la imagen
```bash
docker build -t supermarket-sales-prediction .
```

### Ejecutar el contenedor
```bash
docker run -d --name supermarket-sales -p 8501:8501 -v ./models:/app/models -v ./data:/app/data supermarket-sales-prediction
```

### Ver logs del contenedor
```bash
docker logs supermarket-sales
```

### Entrar al contenedor para depuración
```bash
docker exec -it supermarket-sales bash
```

### Detener y eliminar el contenedor
```bash
docker stop supermarket-sales
docker rm supermarket-sales
```

## Próximos Pasos

- Configurar CI/CD para automatizar el proceso de construcción y despliegue
- Implementar monitoreo de la aplicación
- Optimizar la imagen Docker para reducir su tamaño
- Configurar respaldo automático de datos y modelos
