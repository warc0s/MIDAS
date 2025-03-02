# Midas Touch

## Descripción General

Midas Touch es el componente que automatiza el proceso completo desde la carga de datos hasta el entrenamiento de modelos. El sistema toma como entrada un dataset y una descripción en lenguaje natural de lo que se desea predecir, y genera automáticamente un modelo entrenado, documentación detallada y métricas de rendimiento.

Este componente utiliza tecnologías de IA, específicamente agentes y modelos de lenguaje grande (LLM) de Gemini (Gemini 2.0 Flash) para interpretar la intención del usuario y guiar el proceso de análisis. Implementa un enfoque basado en múltiples agentes especializados que colaboran para realizar todas las etapas del flujo de trabajo de machine learning.

Midas Touch es, a grandes rasgos, una implementación de agentes construido sobre Python "Vanilla" y bibliotecas estándar de ciencia de datos, destacando por su capacidad de autoorganización y recuperación ante fallos.

## Arquitectura Técnica

### Backend:

El backend de Midas Touch está implementado en Python y utiliza un diseño modular basado en agentes especializados:

- **Framework central**: 
  - `AICortex`: *Clase principal* que coordina el flujo de trabajo completo.
  - `OperationalContext`: *Memoria compartida y centro de coordinación* que mantiene el estado global del workflow y permite a los agentes acceder y modificar información que será utilizada por otros agentes en etapas posteriores. Contiene exactamente los valores:
 
***
    # Etapa actual del workflow
    'current_stage': WorkflowStage.DATA_LOADING,
    
    # Datos del dataset
    'dataset': None,            # Dataset cargado
    'target_column': None,      # Columna objetivo identificada
    'data_statistics': {},      # Estadísticas del dataset (ej. distribuciones, valores nulos, etc.)
    
    # Información del problema a resolver
    'problem_type': None,       # Tipo de problema: clasificación o regresión
    
    # Información del proceso y validaciones
    'validation_reports': {},   # Reportes generados durante la validación del modelo
    'pipeline_steps': [],       # Lista de pasos aplicados en el pipeline de procesamiento
    'model_versions': [],       # Versiones del modelo generadas o actualizadas durante el workflow
    
    # Gestión de errores y contingencias
    'error_log': [],            # Registro de errores ocurridos
    'retry_count': 0,           # Contador de reintentos en caso de fallos
    'fallback_activated': False, # Indicador que señala si se activó el modo fallback
    'performance_metrics': {}   # Métricas de rendimiento del modelo (ej. precisión, recall, etc.)
***

- **Agentes especializados**:
  - `IntentAgent`: *Analiza la descripción del usuario* utilizando un LLM para determinar el objetivo del análisis y el tipo de problema (clasificación/regresión).
  - `DataGuardianAgent`: *Analiza el dataset* e identifica la columna objetivo mencionada explícitamente en el prompt del usuario.
  - `DataAlchemistAgent`: *Realiza la limpieza y transformación de datos* adaptándose al tipo de problema y características de los datos.
  - `ModelShamanAgent`: *Selecciona, entrena y evalúa modelos* automáticamente, con soporte completo para problemas multiclase.
  - `OracleAgent`: *Valida la calidad* del flujo completo y los resultados.
  - `NotebookScribeAgent`: *Documenta todo el proceso* en formato Jupyter Notebook.
  - `PhoenixAgent`: *Implementa recuperación ante fallos* con estrategias adaptativas.

- **Sistema de enumeraciones y tipos de datos**:
  - `WorkflowStage`: Enumera las etapas del workflow (DATA_LOADING, DATA_VALIDATION, FEATURE_ENGINEERING, MODEL_TRAINING, MODEL_VALIDATION, DEPLOYMENT, ERROR_HANDLING).
  - `ErrorSeverity`: Define los niveles de gravedad de errores (INFO, WARNING, CRITICAL, FATAL).
  - `ErrorLogEntry`: Estructura de datos para registrar errores con timestamp, agente, mensaje, nivel, operación, intentos y traceback.
  - `ModelMetadata`: Estructura para metadatos del modelo con información sobre tipo, columna objetivo, características, métricas, estadísticas, transformaciones y mapeo de clases.

- **Tecnologías clave**:
  - Google Generative AI (Gemini): Usado para interpretación semántica y análisis de intención. Usa el LLM Gemini-2.0-Flash.
  - pandas: Para manipulación y análisis de datos.
  - scikit-learn: Para modelos de machine learning y preprocesamiento.
  - joblib: Para serialización de modelos.

- **Flujo de datos**:
  1. El usuario proporciona un dataset y una descripción del objetivo.
  2. IntentAgent extrae directamente la columna objetivo y el tipo de problema (clasificación/regresión) del prompt a través de una consulta al LLM.
  3. DataGuardianAgent identifica y selecciona la columna objetivo mencionada en el dataset.
  4. DataAlchemistAgent preprocesa los datos mediante pipelines adaptativas según el tipo de datos (tratando numéricas y categóricas de forma diferente).
  5. ModelShamanAgent selecciona entre RandomForest o GradientBoosting (según el tipo de problema y características del dataset), lo entrena y valida.
  6. El modelo se serializa junto con metadatos que incluyen el mapeo de clases para problemas de clasificación.
  7. NotebookScribeAgent genera documentación detallada y OracleAgent valida las métricas de rendimiento.

### Frontend:

- **Tecnología**: Streamlit para la interfaz web interactiva.
  
- **Componentes principales**:
  - *Panel de carga de datos*: Para subir archivos CSV, Excel, Parquet o JSON.
  - *Campo de texto*: Para describir la tarea de ML en lenguaje natural.
  - *Visor de dataset*: Muestra una vista previa de los datos cargados.
  - *Panel de métricas*: Visualiza el rendimiento del modelo entrenado.
  - *Visualización de mapeo de clases*: Muestra la correspondencia entre valores numéricos y etiquetas originales.
  - *Sistema de pestañas*: Para navegar entre descargas y logs.
  - *Interfaz de descarga*: Para obtener el modelo, documentación y reportes.

- **Personalización de interfaz**:
  - CSS personalizado para mejorar la experiencia visual.
  - Tarjetas interactivas para métricas y descargas.
  - Estilos universales compatibles con modos claro y oscuro de Streamlit.

## Funcionalidad

Midas Touch ofrece las siguientes capacidades principales:

- **Análisis automático de datasets**:
  - Carga y análisis exploratorio automático de datos.
  - Identificación directa de la columna objetivo mencionada en la descripción del usuario.
  - Detección explícita del tipo de problema (clasificación/regresión) desde el prompt.
  - Validación de calidad de datos y estrategias de mitigación.
  - Análisis de tipos de datos, valores únicos, y valores faltantes por columna.
  - Detección de columnas categóricas con alta cardinalidad (>100 valores únicos).

- **Preprocesamiento adaptativo**:
  - Manejo automático de valores faltantes según el tipo de datos (mediana para numéricas, moda para categóricas).
  - Eliminación de columnas con más del 70% de valores faltantes.
  - Detección y procesamiento de fechas, extrayendo componentes útiles (año, mes, día, día de la semana).
  - Codificación de variables categóricas (OneHotEncoder) y escalado de variables numéricas (StandardScaler).
  - Construcción de pipelines de transformación reproducibles con sklearn.
  - Manejo especial para columnas con formato de fecha detectadas automáticamente.

- **Selección y entrenamiento inteligente de modelos**:
  - Utilización del tipo de problema especificado en el prompt (clasificación/regresión).
  - Soporte robusto para problemas de clasificación multiclase con mapeo automático de etiquetas.
  - Selección entre RandomForest y GradientBoosting según las características del dataset:
    - RandomForest: Para datasets pequeños (<1000 muestras) o con muchas características (>50)
    - GradientBoosting: Para datasets más grandes con pocas características
  - Entrenamiento con validación cruzada (5-fold) para estimaciones robustas.
  - Cálculo de métricas específicas para cada tipo de problema:
    - Clasificación: accuracy, f1 (weighted), precision, recall
    - Regresión: r2, MSE, RMSE
  - Estratificación automática cuando es posible (para problemas de clasificación).
  - Manejo adecuado de clases minoritarias durante la validación.
  - Modelos fallback (DummyClassifier/DummyRegressor) en caso de problemas graves.

- **Documentación y explicabilidad**:
  - Generación de un notebook Jupyter detallando todo el proceso.
  - Documentación paso a paso de cada decisión tomada por el sistema.
  - Inclusión de código reproducible para todas las operaciones.
  - Visualización de métricas y resultados del modelo.
  - Documentación explícita del mapeo entre valores numéricos y etiquetas originales en problemas de clasificación.
  - Organización del notebook por secciones lógicas (carga, exploración, preprocesamiento, entrenamiento, evaluación).
  - Cada etapa incluye tanto explicaciones en markdown como el código Python correspondiente.

- **Recuperación ante fallos**:
  - Sistema resiliente con recuperación automática en diferentes etapas.
  - Decorador `resilient_agent` para funciones con reintentos automáticos y backoff exponencial.
  - Estrategias específicas según el tipo de error detectado:
    - Errores en DataGuardianAgent: Selección de columna alternativa (última columna del dataset)
    - Errores en DataAlchemist: Simplificación del preprocesamiento
    - Errores en ModelShaman: Utilización de modelos fallback más simples
  - Logging detallado para diagnóstico y depuración.
  - Supresión inteligente de advertencias irrelevantes (como UndefinedMetricWarning).
  - Captura y manejo de excepciones en cada etapa crítica.

- **Sistema de logging y seguimiento**:
  - Registro detallado de cada paso del proceso.
  - Estructura multinivel de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL).
  - Captura de stacktraces para errores críticos.
  - Registro temporal de cada operación con timestamps.
  - Guardado de logs tanto en archivo como en UI (interfaz Streamlit).
  - Registro enriquecido con contexto sobre el agente y la operación.

- **Serialización y persistencia**:
  - Guardado del modelo entrenado en formato .joblib.
  - Serialización de metadatos complejos a JSON, con manejo especial para tipos de datos NumPy.
  - Función `convert_to_serializable` para transformar objetos NumPy y otros tipos no serializables.
  - Creación de informes de rendimiento en formato texto.
  - Generación de archivos ZIP con todos los resultados para facilitar la descarga.
  - Versionado de modelos con timestamps.

## Guía de Uso

### Uso desde la interfaz Streamlit:

1. **Inicio de la aplicación**:
   *streamlit run Midas_Touch_Streamlit.py*

2. **Carga de datos**:
   - En el panel lateral, haz clic en "Cargar archivo de datos".
   - Selecciona un archivo en formato CSV, Excel, Parquet o JSON.
   - Se mostrará una vista previa del dataset en el panel principal.
   - También verás un resumen de información sobre las columnas (tipos, valores únicos, valores faltantes).

3. **Descripción de la tarea**:
   - En el campo "Describir tarea de ML", escribe una descripción clara de lo que deseas predecir.
   - **Importante**: Especifica explícitamente la columna objetivo y el tipo de problema.
   - Ejemplos:
     - "Predecir la columna precio de las casas, problema de regresión"
     - "Clasificar clientes según la columna abandono, problema de clasificación"
     - "Determinar si un correo es spam o no en la columna categoría, problema de clasificación"

4. **Iniciar procesamiento**:
   - Haz clic en el botón "Iniciar Procesamiento".
   - El sistema comenzará a analizar los datos y mostrará el progreso en tiempo real.
   - Este proceso puede tomar varios minutos dependiendo del tamaño del dataset.

5. **Revisar resultados**:
   - Una vez completado el proceso, se mostrarán las métricas de rendimiento del modelo.
   - Para problemas de clasificación, se mostrará el mapeo entre valores numéricos y etiquetas originales.
   - Navega por las pestañas para ver:
     - **Descargas**: Opciones para descargar el modelo, notebook y reportes.
     - **Logs**: Registro detallado de todas las operaciones realizadas.

6. **Descargar resultados**:
   - En la pestaña "Descargas", tienes varias opciones:
     - **Todo en uno**: Archivo ZIP con todos los archivos generados.
     - **Notebook**: Documentación en formato .ipynb.
     - **Modelo entrenado**: Archivo .joblib con el modelo serializado.
     - **Reporte de rendimiento**: Métricas detalladas del modelo y mapeo de clases.

### Uso desde línea de comandos:

También puedes utilizar Midas Touch directamente desde la línea de comandos:

*python Midas_Touch_V2_CLI.py*

El sistema te pedirá una descripción de la tarea de ML y procesará el archivo de datos configurado en `CONFIG['DATA_FILE']`. Al finalizar, mostrará un resumen en la consola y guardará todos los archivos generados en las ubicaciones especificadas en CONFIG.

### Configuración del sistema:

El sistema incluye un diccionario `CONFIG` con los siguientes parámetros ajustables:

- `API_KEY_ENV_VAR`: Nombre de la variable de entorno para la API key de Google.
- `MODEL_NAME`: Modelo de Gemini a utilizar (por defecto, 'gemini-2.0-flash').
- `LOG_FILE`: Ruta del archivo de log.
- `LOG_LEVEL`: Nivel de logging (INFO, DEBUG, etc.).
- `DATA_FILE`: Archivo de datos predeterminado.
- `MODEL_DIR`: Directorio para guardar modelos.
- `NOTEBOOK_FILE`: Ruta del notebook generado.
- `RETRIES`: Número de reintentos para diferentes operaciones.
- `MIN_ROWS`: Mínimo de filas recomendado para el dataset.
- `MAX_MISSING_RATIO`: Ratio máximo permitido de valores faltantes.
- `MIN_FEATURE_VARIANCE`: Varianza mínima requerida para características.
- `DEFAULT_TEST_SIZE`: Tamaño predeterminado del conjunto de prueba.
- `RANDOM_SEED`: Semilla para reproducibilidad.
- `PERFORMANCE_THRESHOLDS`: Umbrales mínimos de rendimiento para modelos.

### Ejemplos de entrada/salida:

**Entrada**:
- Dataset: archivo CSV con datos de clientes de un banco
- Descripción: "Predecir si un cliente abandonará el servicio en la columna churn, problema de clasificación"

**Salida**:
- Modelo de clasificación (RandomForest o GradientBoosting) serializado como .joblib
- Metadatos con mapeo de clases (ej: 0 → "No", 1 → "Sí")
- Notebook con documentación detallada del proceso
- Métricas como accuracy, precision, recall y F1-score (weighted para multiclase)
- Reportes en formato texto y JSON con detalles del modelo
- Archivo ZIP con todos los resultados

Durante el proceso, se ofrece información en tiempo real sobre:
- Etapa actual del workflow
- Progreso del procesamiento
- Alertas y mensajes de validación

## Referencias y Recursos

- **Código fuente**:
  - [Midas_Touch_V2_CLI.py](https://github.com/warc0s/MIDAS/blob/main/4midas_touch/Midas_Touch_V2_CLI.py) - Implementación principal
  - [Midas_Touch_Streamlit.py](https://github.com/warc0s/MIDAS/blob/main/4midas_touch/Midas_Touch_Streamlit.py) - Interfaz web

- **Tecnologías principales utilizadas**:
  - [Google Generative AI (Gemini)](https://ai.google.dev/docs) - Para las llamadas a Gemini Flash
  - [scikit-learn](https://scikit-learn.org/) - Para trabajar con los modelos de machine learning
  - [pandas](https://pandas.pydata.org/) - Para la manipulación de datos
  - [Streamlit](https://streamlit.io/) - Para la interfaz web
  - [joblib](https://joblib.readthedocs.io/) - Para serialización de modelos

- **Documentación relacionada**:
  - [Sklearn Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
  - [Pandas DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html)
  - [Jupyter Notebook Format](https://nbformat.readthedocs.io/en/latest/)
  - [Streamlit Components](https://docs.streamlit.io/library/components)

## Limitaciones Actuales

- **Soporte de modelos ML**: Actualmente solo implementa modelos de scikit-learn, específicamente RandomForest y GradientBoosting (no usa búsqueda de hiperparámetros).
- **Soporte de modelos LLM**: Ahora mismo usa exclusivamente Gemini 2.0 Flash. En un futuro, podría usarse LiteLLM y definir el modelo + api_key en el .env.
- **Tamaño de datasets**: Está optimizado para datasets de tamaño pequeño a mediano (recomendable hasta ~25K filas). Datasets muy grandes pueden causar problemas de rendimiento.
- **Complejidad de intención**: Aunque el sistema extrae directamente la columna objetivo y el tipo de problema del prompt, descripciones ambiguas pueden llevar a interpretaciones incorrectas.
- **Preprocesamiento especializado**: Algunas transformaciones de dominio específico (como procesamiento avanzado de texto, embeddings, o series temporales) no están implementadas.
- **Explicabilidad de modelos**: No incluye herramientas avanzadas de interpretabilidad como SHAP o LIME.
- **Modo interactivo**: No implementa un modo "semi-manual" donde el sistema consulte al usuario sobre decisiones clave (ej: tratamiento de outliers, imputación de valores).
- **Visualizaciones**: En el notebook generado no se incluyen gráficas que podrían ser relevantes (importancia de características, matriz de correlación, etc.).
- **Umbrales predeterminados**: Los umbrales de rendimiento y otros parámetros están codificados en CONFIG y no son ajustables dinámicamente desde la interfaz streamlit.
- **Validación de entrada**: No hay validación avanzada del texto introducido por el usuario, lo que puede afectar la interpretación si no se sigue el formato recomendado.
- **Limitaciones de robustez**: Puede tener dificultades con estructuras de datos muy complejas o tipos de datos no estándar.

## Capturas:

![Midas Imagen](https://github.com/warc0s/MIDAS/blob/main/Extra/Imagenes/Midas_Touch_Interfaz_6_0.png?raw=true)

![Midas Imagen](https://github.com/warc0s/MIDAS/blob/main/Extra/Imagenes/Midas_Touch_Preprocesamiento_3_2.png?raw=true)