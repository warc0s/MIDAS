# Midas Touch

## Descripción General

Midas Touch es el componente central de MIDAS, el cual automatiza el proceso completo desde la carga de datos hasta el entrenamiento del modelo. El sistema toma como entrada un dataset y una descripción en lenguaje natural de lo que se desea predecir, y genera automáticamente un modelo entrenado, documentación detallada y métricas de rendimiento.

Este componente utiliza tecnologías de IA, específicamente agentes y modelos de lenguaje grande (LLM) de Gemini (Gemini 2.0 Flash) para interpretar la intención del usuario y guiar el proceso de análisis. Implementa un enfoque basado en múltiples agentes especializados que colaboran para realizar todas las etapas del flujo de trabajo de machine learning.

Midas Touch es, a grandes rasgos, una implementación de agentes construido sobre Python "Vanilla" y bibliotecas estándar de ciencia de datos, destacando por su capacidad de autoorganización y recuperación ante fallos.

## Arquitectura Técnica

### Backend:

El backend de Midas Touch está implementado en Python y utiliza un diseño modular basado en agentes especializados:

- **Framework central**: 
  - `AICortex`: *Clase principal* que coordina el flujo de trabajo completo.
  - `OperationalContext`: *Memoria compartida y centro de coordinación* que mantiene el estado global del workflow y permite a los agentes acceder y modificar información que será utilizada por otros agentes en etapas posteriores.

- **Agentes especializados**:
  - `IntentAgent`: *Analiza la descripción del usuario* utilizando un LLM para determinar el objetivo del análisis y el tipo de problema (clasificación/regresión).
  - `DataGuardianAgent`: *Analiza el dataset* e identifica la columna objetivo mencionada explícitamente en el prompt del usuario.
  - `DataAlchemistAgent`: *Realiza la limpieza y transformación de datos* adaptándose al tipo de problema y características de los datos.
  - `ModelShamanAgent`: *Selecciona, entrena y evalúa modelos* automáticamente, con soporte completo para problemas multiclase.
  - `OracleAgent`: *Valida la calidad* del flujo completo y los resultados.
  - `NotebookScribeAgent`: *Documenta todo el proceso* en formato Jupyter Notebook.
  - `PhoenixAgent`: *Implementa recuperación ante fallos* con estrategias adaptativas.

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
  - *Sistema de pestañas*: Para navegar entre notebook, descargas y logs.
  - *Interfaz de descarga*: Para obtener el modelo, documentación y reportes.

## Funcionalidad

Midas Touch ofrece las siguientes capacidades principales:

- **Análisis automático de datasets**:
  - Carga y análisis exploratorio automático de datos.
  - Identificación directa de la columna objetivo mencionada en la descripción del usuario.
  - Detección explícita del tipo de problema (clasificación/regresión) desde el prompt.
  - Validación de calidad de datos y estrategias de mitigación.

- **Preprocesamiento adaptativo**:
  - Manejo automático de valores faltantes según el tipo de datos (mediana para numéricas, moda para categóricas).
  - Eliminación de columnas con más del 70% de valores faltantes.
  - Detección y procesamiento de fechas, extrayendo componentes útiles (año, mes, día, día de la semana).
  - Codificación de variables categóricas (OneHotEncoder) y escalado de variables numéricas (StandardScaler).
  - Construcción de pipelines de transformación reproducibles.

- **Selección y entrenamiento inteligente de modelos**:
  - Utilización del tipo de problema especificado en el prompt (clasificación/regresión).
  - Soporte robusto para problemas de clasificación multiclase con mapeo automático de etiquetas.
  - Selección entre RandomForest y GradientBoosting según las características del dataset:
    - RandomForest: Para datasets pequeños (<1000 muestras) o con muchas características (>50)
    - GradientBoosting: Para datasets más grandes con pocas características
  - Entrenamiento con validación cruzada (5-fold) para estimaciones robustas.
  - Cálculo de métricas de rendimiento adecuadas para cada tipo de problema.
  - Estratificación automática cuando es posible (para problemas de clasificación).

- **Documentación y explicabilidad**:
  - Generación de un notebook Jupyter detallando todo el proceso.
  - Documentación paso a paso de cada decisión tomada por el sistema.
  - Inclusión de código reproducible para todas las operaciones.
  - Visualización de métricas y resultados del modelo.
  - Documentación explícita del mapeo entre valores numéricos y etiquetas originales en problemas de clasificación.

- **Recuperación ante fallos**:
  - Sistema resiliente con recuperación automática en diferentes etapas.
  - Estrategias específicas según el tipo de error detectado:
    - Errores en DataGuardianAgent: Selección de columna alternativa
    - Errores en DataAlchemist: Simplificación del preprocesamiento
    - Errores en ModelShaman: Utilización de modelos fallback más simples
  - Logging detallado para diagnóstico y depuración.
  - Supresión inteligente de advertencias irrelevantes para mejorar la experiencia del usuario.

## Guía de Uso

### Uso desde la interfaz Streamlit:

1. **Inicio de la aplicación**:
   *streamlit run Midas_Touch_Streamlit.py*

2. **Carga de datos**:
   - En el panel lateral, haz clic en "Cargar archivo de datos".
   - Selecciona un archivo en formato CSV, Excel, Parquet o JSON.
   - Se mostrará una vista previa del dataset en el panel principal.

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
     - **Notebook**: Documentación detallada del proceso en formato Jupyter.
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

El sistema te pedirá una descripción de la tarea de ML y procesará el archivo de datos configurado en `CONFIG['DATA_FILE']`.

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

- **Documentación relacionada**:
  - [Sklearn Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
  - [Pandas DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html)
  - [Jupyter Notebook Format](https://nbformat.readthedocs.io/en/latest/)

## Limitaciones Actuales

- **Soporte de modelos ML**: Actualmente solo implementa modelos de scikit-learn, específicamente RandomForest y GradientBoosting (no usa búsqueda de hiperparámetros).
- **Soporte de modelos LLM**: Ahora mismo usa exclusivamente Gemini 2.0 Flash. En un futuro, podría usarse LiteLLM y definir el modelo + api_key en el .env.
- **Tamaño de datasets**: Está optimizado para datasets de tamaño pequeño a mediano (recomendable hasta ~25K filas). Datasets muy grandes pueden causar problemas de rendimiento.
- **Complejidad de intención**: Aunque el sistema extrae directamente la columna objetivo y el tipo de problema del prompt, descripciones ambiguas pueden llevar a interpretaciones incorrectas.
- **Preprocesamiento especializado**: Algunas transformaciones de dominio específico (como procesamiento avanzado de texto, embeddings, o series temporales) no están implementadas.
- **Explicabilidad de modelos**: No incluye herramientas avanzadas de interpretabilidad como SHAP o LIME.
- **Modo interactivo**: No implementa un modo "semi-manual" donde el sistema consulte al usuario sobre decisiones clave (ej: tratamiento de outliers, imputación de valores).
- **Visualizaciones**: En el notebook generado no se incluyen gráficas que podrían ser relevantes (importancia de características, matriz de correlación, etc.).
