# Función para cargar resultados de modelos
@st.cache_data
def load_model_results():
    results = {}
    # Definimos las rutas de los resultados
    result_paths = {
        'MLP': 'models/mlp/results/mlp_results.joblib',
        'LSTM': 'models/lstm/results/lstm_results.joblib',
        'CNN': 'models/cnn/results/cnn_results.joblib',
        'Baseline': 'models/baseline/results/baseline_results.joblib',
        'Híbrido': 'models/hybrid/results/hybrid_results.joblib',
        'Ensemble': 'models/ensemble/results/ensemble_results.joblib'
    }
    
    # Verificar si existe al menos un archivo de resultados
    results_found = any(os.path.exists(path) for path in result_paths.values())
    
    if results_found:
        # Cargar los resultados de modelos entrenados
        for name, path in result_paths.items():
            try:
                if os.path.exists(path):
                    results[name] = joblib.load(path)
                    st.sidebar.success(f"Resultados del modelo {name} cargados correctamente")
                else:
                    st.sidebar.warning(f"No se encontraron resultados para el modelo {name}")
            except Exception as e:
                st.sidebar.warning(f"No se pudieron cargar los resultados del modelo {name}: {e}")
                
        # Crear datos para comparación
        model_names = []
        mae_values = []
        mse_values = []
        rmse_values = []
        r2_values = []
        
        for name, model_results in results.items():
            if name != 'Baseline':
                if 'metrics' in model_results:
                    model_names.append(name)
                    mae_values.append(model_results['metrics']['mae'])
                    mse_values.append(model_results['metrics']['mse'])
                    rmse_values.append(model_results['metrics']['rmse'])
                    r2_values.append(model_results['metrics']['r2'])
            elif 'rf_optimized' in model_results:
                model_names.append('Baseline')
                mae_values.append(model_results['rf_optimized']['metrics']['mae'])
                mse_values.append(model_results['rf_optimized']['metrics']['mse'])
                rmse_values.append(model_results['rf_optimized']['metrics']['rmse'])
                r2_values.append(model_results['rf_optimized']['metrics']['r2'])
        
        # Crear DataFrame de comparación
        if model_names:
            results['comparison'] = pd.DataFrame({
                'Modelo': model_names,
                'MAE': mae_values,
                'MSE': mse_values,
                'RMSE': rmse_values,
                'R²': r2_values
            })
        
        st.sidebar.success("Resultados de modelos cargados correctamente")
    else:
        # Creamos resultados de ejemplo para modo demostración
        st.sidebar.warning("No se encontraron resultados de modelos. Usando resultados de demostración.")
        
        # Resultados de ejemplo para cada modelo
        models = ['MLP', 'LSTM', 'CNN', 'Baseline', 'Híbrido', 'Ensemble']
        for model in models:
            results[model] = {
                'metrics': {
                    'mae': np.random.uniform(10, 50),
                    'mse': np.random.uniform(100, 500),
                    'rmse': np.random.uniform(15, 40),
                    'r2': np.random.uniform(0.7, 0.95)
                },
                'predictions': np.random.normal(100, 20, size=50),
                'actual': np.random.normal(100, 20, size=50)
            }
        
        # Resultados para comparación
        results['comparison'] = pd.DataFrame({
            'Modelo': models,
            'MAE': [results[m]['metrics']['mae'] for m in models],
            'MSE': [results[m]['metrics']['mse'] for m in models],
            'RMSE': [results[m]['metrics']['rmse'] for m in models],
            'R²': [results[m]['metrics']['r2'] for m in models]
        })
    
    return results
