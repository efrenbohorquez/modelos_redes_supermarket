# Proyecto de Análisis y Predicción de Ventas de Supermercado

Este repositorio contiene un proyecto completo para el análisis y predicción de ventas de supermercado utilizando múltiples modelos de redes neuronales y técnicas de aprendizaje automático.

## Descripción

El proyecto implementa un sistema completo de análisis y predicción de ventas para supermercados, utilizando diversos modelos de machine learning y deep learning. La aplicación web desarrollada con Streamlit permite visualizar datos, comparar modelos y realizar predicciones interactivamente.

### Modelos implementados

- **MLP (Red Neuronal Multicapa)**: Modelo básico de red neuronal para datos tabulares.
- **LSTM (Long Short-Term Memory)**: Red neuronal recurrente para análisis de series temporales.
- **CNN (Red Neuronal Convolucional)**: Red neuronal para patrones visuales.
- **Baseline (Random Forest)**: Modelo clásico de machine learning como referencia.
- **Modelo Híbrido**: Combinación de diferentes tipos de redes neuronales.
- **Ensemble (Stacking)**: Meta-modelo que combina las predicciones de los modelos anteriores.

## Estructura del Proyecto

```
├── data/                     # Datos
│   ├── supermarket_sales.xlsx # Archivo original de datos
│   ├── supermarket_sales.csv  # Datos convertidos a CSV
│   └── processed/            # Datos procesados para cada modelo
├── models/                   # Modelos entrenados
│   ├── mlp/                  # Modelo MLP
│   │   ├── mlp_ventas_totales.h5 # Modelo guardado
│   │   └── results/         # Resultados del entrenamiento
│   ├── lstm/                 # Modelo LSTM
│   │   ├── lstm_ventas_totales_seq7.h5 # Modelo guardado
│   │   └── results/         # Resultados del entrenamiento 
│   ├── cnn/                  # Modelo CNN
│   │   ├── cnn_ventas_totales.h5 # Modelo guardado
│   │   └── results/         # Resultados del entrenamiento
│   ├── baseline/             # Modelos baseline (Random Forest)
│   │   ├── models/          # Modelos básico y optimizado
│   │   └── results/         # Resultados del entrenamiento
│   ├── hybrid/               # Modelo híbrido (MLP+LSTM+CNN)
│   │   └── results/         # Resultados del modelo híbrido
│   └── ensemble/             # Modelo ensemble (Stacking)
│       └── results/         # Resultados del ensemble
├── app.py                    # Aplicación principal en Streamlit
├── train_models.py           # Script para entrenar todos los modelos
├── data_utils.py             # Utilidades para procesamiento de datos
├── model_results_loader.py   # Cargador de resultados para la visualización
├── visualization_utils.py    # Utilidades para visualización de datos
├── documentation_utils.py    # Utilidades para documentación
├── requirements.txt          # Dependencias del proyecto
├── academic_report.md        # Informe académico detallado
├── web_publication.md        # Publicación web para dueños de negocios
└── todo.md                   # Lista de tareas pendientes
```

## Jupyter Notebooks (Proceso de Desarrollo)

- `01_exploratory_analysis.ipynb` - Análisis exploratorio de datos
- `02_data_preprocessing.ipynb` - Preprocesamiento de datos
- `03_model_mlp.ipynb` - Desarrollo del modelo MLP
- `04_model_lstm.ipynb` - Desarrollo del modelo LSTM
- `05_model_cnn.ipynb` - Desarrollo del modelo CNN
- `06_model_baseline.ipynb` - Desarrollo del modelo baseline
- `07_model_hybrid_ensemble.ipynb` - Desarrollo de modelos híbrido y ensemble

## Instalación y Uso

### Requisitos previos

- Python 3.8+
- Tensorflow 2.x
- Pandas, NumPy, Scikit-learn
- Streamlit
- Matplotlib, Seaborn, Plotly

### Instrucciones de instalación

1. Clonar el repositorio:
```bash
git clone https://github.com/efrenbohorquez/modelos_redes_supermarket.git
cd modelos_redes_supermarket
```

2. Instalar las dependencias:
```bash
pip install -r requirements.txt
```

### Ejecución

Para entrenar los modelos:
```bash
python train_models.py
```

Para ejecutar la aplicación web:
```bash
streamlit run app.py
```

## Características de la aplicación

La aplicación Streamlit proporciona una interfaz interactiva para:
- Explorar los datos de ventas
- Realizar predicciones con diferentes modelos
- Comparar el rendimiento de los modelos
- Visualizar resultados con gráficos interactivos
- Exportar resultados y análisis

## Licencia

Este proyecto está bajo la licencia MIT. Ver el archivo `LICENSE` para más detalles.
│   └── 07_model_hybrid_ensemble.ipynb
├── utils/                    # Utilidades
│   ├── data_utils.py         # Utilidades para procesamiento de datos
│   ├── visualization_utils.py # Utilidades para visualización
│   ├── documentation_utils.py # Utilidades para documentación
│   └── export_utils.py       # Utilidades para exportación
├── requirements.txt          # Dependencias del proyecto
├── README.md                 # Este archivo
└── todo.md                   # Lista de tareas del proyecto
```

## Requisitos

Para ejecutar este proyecto, necesitarás Python 3.8+ y las siguientes dependencias:

```
numpy==1.24.3
pandas==2.0.3
matplotlib==3.7.2
seaborn==0.12.2
scikit-learn==1.3.0
tensorflow==2.13.0
keras==2.13.1
streamlit==1.24.0
plotly==5.15.0
openpyxl==3.1.2
joblib==1.3.1
fpdf2==2.7.4
```

Puedes instalar todas las dependencias con:

```bash
pip install -r requirements.txt
```

## Notebooks

El proyecto incluye los siguientes notebooks:

1. **01_exploratory_analysis.ipynb**: Análisis exploratorio de los datos de ventas.
2. **02_data_preprocessing.ipynb**: Preprocesamiento y preparación de datos para modelado.
3. **03_model_mlp.ipynb**: Implementación y entrenamiento de Red Neuronal Multicapa (MLP).
4. **04_model_lstm.ipynb**: Implementación y entrenamiento de Red Neuronal Recurrente LSTM.
5. **05_model_cnn.ipynb**: Implementación y entrenamiento de Red Neuronal Convolucional (CNN).
6. **06_model_baseline.ipynb**: Implementación y entrenamiento de modelos baseline (Random Forest, etc.).
7. **07_model_hybrid_ensemble.ipynb**: Implementación y entrenamiento de modelos híbridos y ensemble.

## Aplicación Streamlit

La aplicación Streamlit proporciona una interfaz interactiva para:

- Explorar los datos de ventas
- Realizar predicciones con diferentes modelos
- Comparar el rendimiento de los modelos
- Visualizar resultados con gráficos interactivos
- Exportar resultados y análisis

Para ejecutar la aplicación:

```bash
cd app
streamlit run app.py
```

## Modelos Implementados

El proyecto implementa seis modelos de redes neuronales:

1. **Red Neuronal Multicapa (MLP)**: Para predicción basada en características tabulares.
2. **Red Neuronal Recurrente LSTM**: Para capturar patrones temporales en las ventas.
3. **Red Neuronal Convolucional (CNN)**: Para detectar patrones locales en los datos.
4. **Modelos Baseline**: Random Forest y otros modelos tradicionales como punto de referencia.
5. **Modelo Híbrido**: Combinación de MLP, LSTM y CNN para aprovechar las fortalezas de cada arquitectura.
6. **Modelo Ensemble**: Combinación de múltiples modelos para mejorar la robustez y precisión.

## Documentación

El proyecto incluye dos documentos principales:

1. **Informe Académico**: Un informe detallado tipo tesis de maestría con la metodología, resultados y hallazgos.
2. **Publicación Web**: Una publicación orientada a dueños de supermercados con recomendaciones prácticas.

## Contribuciones

Las contribuciones son bienvenidas. Por favor, sigue estos pasos:

1. Haz fork del repositorio
2. Crea una rama para tu característica (`git checkout -b feature/nueva-caracteristica`)
3. Haz commit de tus cambios (`git commit -am 'Añadir nueva característica'`)
4. Haz push a la rama (`git push origin feature/nueva-caracteristica`)
5. Crea un nuevo Pull Request

## Licencia

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo LICENSE para más detalles.

## Implementación con Docker

Este proyecto está containerizado usando Docker para facilitar su despliegue y ejecución. Sigue estos pasos para ejecutar la aplicación usando Docker:

### Prerrequisitos

- [Docker](https://www.docker.com/get-started) instalado en tu sistema
- [Docker Compose](https://docs.docker.com/compose/install/) (opcional, para usar docker-compose.yml)

### Instalación y ejecución

#### Opción 1: Usando scripts de automatización

**Para Windows (PowerShell):**
```powershell
.\build_and_run.ps1
```

**Para Linux/Mac (Bash):**
```bash
chmod +x build_and_run.sh
./build_and_run.sh
```

#### Opción 2: Usando Docker Compose

```bash
docker-compose up -d
```

#### Opción 3: Comandos Docker manuales

```bash
# Construir la imagen
docker build -t supermarket-sales-prediction .

# Ejecutar el contenedor
docker run -d --name supermarket-sales -p 8501:8501 -v ./models:/app/models -v ./data:/app/data supermarket-sales-prediction
```

### Acceder a la aplicación

Una vez que el contenedor esté en funcionamiento, accede a la aplicación web en tu navegador:
- URL: `http://localhost:8501`

### Solución de problemas comunes

- **Error de carga de modelos**: Verifica que los archivos de modelos (.h5 y .joblib) existan en las carpetas correspondientes.
- **Error de permisos de volumen**: Asegúrate de que Docker tenga permisos para acceder a las carpetas de modelos y datos.
- **Problemas de memoria**: Si Docker se queda sin memoria, intenta aumentar la memoria asignada en la configuración de Docker.

## Contacto

Para preguntas o comentarios, por favor contacta a [tu-email@ejemplo.com].
