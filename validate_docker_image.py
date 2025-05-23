#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script para verificar que la imagen Docker se construya correctamente.
Ejecutar después de construir la imagen para validar su contenido.
"""

import subprocess
import sys
import re
import os
import time

def run_command(command):
    """Ejecuta un comando en la terminal y devuelve su salida."""
    print(f"Ejecutando: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout, result.stderr, result.returncode

def check_docker_installed():
    """Comprueba si Docker está instalado y funcionando."""
    print("1. Verificando la instalación de Docker...")
    stdout, stderr, returncode = run_command("docker --version")
    if returncode != 0:
        print("❌ Docker no está instalado o no está disponible en el PATH.")
        print(f"Error: {stderr}")
        return False
    print(f"✅ Docker instalado: {stdout.strip()}")
    return True

def check_image_exists(image_name):
    """Comprueba si la imagen Docker existe."""
    print(f"\n2. Verificando si existe la imagen {image_name}...")
    stdout, stderr, returncode = run_command(f"docker images -q {image_name}")
    if stdout.strip():
        print(f"✅ Imagen {image_name} encontrada.")
        return True
    print(f"❌ Imagen {image_name} no encontrada.")
    return False

def check_container_runs(image_name):
    """Intenta ejecutar un contenedor con la imagen y verifica que funcione."""
    container_name = f"test-{image_name}-{int(time.time())}"
    print(f"\n3. Intentando ejecutar un contenedor de prueba con la imagen {image_name}...")
    
    # Ejecutar el contenedor en modo detached
    stdout, stderr, returncode = run_command(
        f"docker run -d --name {container_name} -p 9501:8501 {image_name}"
    )
    
    if returncode != 0:
        print(f"❌ No se pudo iniciar el contenedor: {stderr}")
        return False
    
    container_id = stdout.strip()
    print(f"✅ Contenedor iniciado con ID: {container_id}")
    
    # Esperar a que el contenedor se inicie completamente
    print("   Esperando 10 segundos para que el contenedor se inicie...")
    time.sleep(10)
    
    # Verificar el estado del contenedor
    stdout, stderr, returncode = run_command(f"docker ps -f id={container_id} --format '{{{{.Status}}}}'")
    
    if "Up" not in stdout:
        print(f"❌ El contenedor no está en ejecución. Estado: {stdout.strip()}")
        cleanup_container(container_name)
        return False
    
    print(f"✅ El contenedor está en ejecución. Estado: {stdout.strip()}")
    
    # Verificar los logs del contenedor
    stdout, stderr, returncode = run_command(f"docker logs {container_id}")
    
    if "Server started" in stdout or "Streamlit app running" in stdout or "You can now view" in stdout:
        print("✅ La aplicación Streamlit parece estar ejecutándose correctamente.")
    else:
        print("⚠️ No se pudo confirmar que Streamlit esté ejecutándose. Revisa los logs:")
        print(stdout[-500:] if len(stdout) > 500 else stdout)
    
    # Limpiar el contenedor de prueba
    cleanup_container(container_name)
    return True

def cleanup_container(container_name):
    """Detiene y elimina el contenedor de prueba."""
    print(f"\n4. Limpiando el contenedor de prueba {container_name}...")
    run_command(f"docker stop {container_name}")
    run_command(f"docker rm {container_name}")
    print(f"✅ Contenedor {container_name} eliminado.")

def main():
    image_name = "supermarket-sales-prediction"
    
    print("=================================================")
    print(" VALIDADOR DE IMAGEN DOCKER PARA LA APLICACIÓN")
    print("=================================================")
    
    if not check_docker_installed():
        return 1
    
    if not check_image_exists(image_name):
        print("\n⚠️ La imagen no existe. ¿Deseas construirla ahora? (s/n)")
        response = input("> ").lower()
        if response == "s":
            print("\nConstruyendo imagen Docker...")
            stdout, stderr, returncode = run_command("docker build -t supermarket-sales-prediction .")
            if returncode != 0:
                print(f"❌ Error al construir la imagen: {stderr}")
                return 1
            print("✅ Imagen construida correctamente.")
        else:
            print("\nPor favor, construye la imagen primero con 'docker build -t supermarket-sales-prediction .'")
            return 1
    
    check_container_runs(image_name)
    
    print("\n=================================================")
    print(" VALIDACIÓN COMPLETA")
    print("=================================================")
    print("Para iniciar la aplicación, ejecuta uno de los siguientes comandos:")
    print("\nOpción 1 (scripts de automatización):")
    print("  - Windows: .\\build_and_run.ps1")
    print("  - Linux/Mac: ./build_and_run.sh")
    print("\nOpción 2 (Docker Compose):")
    print("  docker-compose up -d")
    print("\nOpción 3 (Docker CLI):")
    print("  docker run -d --name supermarket-sales -p 8501:8501 -v ./models:/app/models -v ./data:/app/data supermarket-sales-prediction")
    print("\nLuego accede a http://localhost:8501 en tu navegador.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
