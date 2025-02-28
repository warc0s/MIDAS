# Componente Midas Test

## Descripción General

MIDAS Test es un componente especializado en la evaluación exhaustiva de modelos de machine learning almacenados en formato joblib. Su propósito principal es analizar la calidad, rendimiento y robustez de modelos ML mediante una arquitectura de agentes colaborativos basados en IA.

El sistema utiliza tecnología LLM (Large Language Models) para coordinar múltiples agentes especializados que evalúan diferentes aspectos de los modelos ML. MIDAS Test implementa un enfoque de colaboración multi-agente donde cada agente aporta su perspectiva especializada para generar un informe completo.

MIDAS Test se basa en el framework AG2 para la gestión de agentes conversacionales y utiliza Streamlit para proporcionar una interfaz de usuario accesible.

## Arquitectura Técnica

### Backend:

- **Lenguaje y Frameworks:** 
  - Python 3.x
  - AG2 para la gestión de agentes IA
  - Scikit-learn para manipulación de modelos ML
  - Joblib para carga/guardado de modelos

- **Componentes Clave:**
  - *Agentes Especializados*:
    - **Model Analyzer**: Examina la estructura y características del modelo ML.
    - **Performance Tester**: Evalúa métricas de rendimiento como latencia, uso de memoria y CPU.
    - **Robustness Checker**: Verifica la resistencia del modelo ante entradas anómalas.
    - **Output Validator**: Confirma la validez y formato de las predicciones del modelo.
  
  - *Gestor de Comunicación*:
    - **GroupChat**: Facilita la comunicación entre agentes.
    - **GroupChatManager**: Coordina el flujo de la conversación y turno de los agentes.
  
  - *Modelo LLM Base*:
    - Utiliza *meta-llama/Llama-3.3-70B-Instruct-Turbo* a través de la API de DeepInfra.
    - Configuración personalizable de temperatura y seed para resultados consistentes.
  
  - *Módulos de Procesamiento*:
    - **load_model**: Carga modelos joblib y mide tiempo de carga.
    - **measure_latency**: Evalúa tiempos de respuesta en diferentes tamaños de batch.
    - **check_robustness**: Prueba comportamiento ante valores nulos, extremos y tipos incorrectos.
    - **validate_predictions**: Verifica la consistencia y formato de las predicciones.
    - **generate_markdown_report**: Compila los hallazgos en formato Markdown estructurado.

- **Flujo de Procesamiento**:
  1. Carga del modelo joblib.
  2. Validación inicial del modelo (compatibilidad con Scikit-learn).
  3. Generación de datos de muestra para pruebas.
  4. Ejecución paralela de pruebas de rendimiento, robustez y validación.
  5. Compilación de métricas y resultados.
  6. Activación de agentes IA para análisis especializado.
  7. Generación de informe final en formato Markdown.

### Frontend:

- **Tecnologías:**
  - Streamlit para la interfaz web interactiva
  - Componentes UI de Streamlit: file_uploader, expanders, download_button

- **Estructura de la Interfaz:**
  - Sección de carga de archivos
  - Panel de progreso y estado
  - Visualización de resultados en secciones expandibles
  - Botones para iniciar evaluación y descargar informes

## Funcionalidad

- **Análisis Profundo de Modelos ML**: Evalúa múltiples aspectos del modelo incluyendo validez, rendimiento y robustez.

- **Métricas de Rendimiento**: 
  - Tiempo de carga del modelo
  - Uso de memoria durante predicciones
  - Utilización de CPU
  - Latencia en diferentes tamaños de batch
  - Throughput (predicciones por segundo)

- **Pruebas de Robustez**:
  - Comportamiento ante valores nulos
  - Resistencia a valores fuera de rango
  - Manejo de tipos de datos incorrectos
  - Consistencia de predicciones

- **Validación de Salidas**:
  - Verificación de formato correcto
  - Validación de rangos esperados
  - Comprobación de probabilidades (cuando aplica)

- **Recomendación Automatizada**: Clasificación del modelo como "APTO" o "NO APTO" basada en criterios objetivos.

- **Reporte Markdown**: Generación automática de documentación estructurada con los hallazgos y recomendaciones.

## Guía de Uso

### A través de la Interfaz Web (Streamlit):

1. Inicie la aplicación ejecutando:
   *streamlit run app.py*

2. En la interfaz web, haga clic en "📂 Carga un archivo .joblib" y seleccione el modelo a evaluar.

3. Una vez cargado el modelo, pulse el botón "🔄 Iniciar Evaluación con los Agentes" para comenzar el análisis.

4. El sistema mostrará el progreso del análisis en tiempo real.

5. Al finalizar, pulse "📄 Finalizar Análisis y Descargar Reporte" para obtener los resultados.

6. Explore los resultados en las secciones expandibles:
   - "📌 Información del Modelo": Datos básicos como tiempo de carga y tamaño
   - "📈 Métricas de Rendimiento": Detalles sobre uso de recursos
   - "⚠️ Pruebas de Robustez": Resultados de las pruebas de resistencia

7. Descargue el informe completo en formato Markdown utilizando el botón "⬇️ Descargar Reporte".

### Mediante Línea de Comandos:

1. Ejecute el script principal:
   *python agents_test.py*

2. Cuando se solicite, ingrese la ruta completa al archivo joblib que desea analizar.

3. El sistema ejecutará automáticamente todas las pruebas y generará un informe en el archivo "model_analysis_report.md".

### Ejemplo de Salida:

El reporte generado contendrá secciones como:

# 📊 Model Analysis Report
**Generated on:** 2025-02-28 15:30:45

---

## 🔍 Model Overview
[Información general sobre el modelo y sus características]

## ⚙️ Performance Metrics
[Detalles sobre rendimiento, memoria y CPU]

## ⏳ Latency Analysis
[Análisis de tiempos de respuesta]

## ✅ Predictions Validity
[Validación de las salidas del modelo]

## 🛡️ Robustness Tests
[Resultados de pruebas de resistencia]

## 📌 Final Recommendation
**APTO**

## 🔧 Suggested Improvements
[Recomendaciones para mejorar el modelo]

## Referencias y Recursos

- **Frameworks Utilizados**:
  - AG2: [https://docs.ag2.ai/docs/user-guide/basic-concepts/installing-ag2](https://docs.ag2.ai/docs/user-guide/basic-concepts/installing-ag2)
  - Streamlit: [https://streamlit.io/](https://streamlit.io/)
  - DeepInfra API: [https://deepinfra.com/](https://deepinfra.com/)

- **Modelos LLM**:
  - Llama 3.3 70B: https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct

## Limitaciones Actuales

- El componente está optimizado para modelos Scikit-learn y puede tener limitaciones con otros frameworks de ML.
- Las pruebas de robustez son algo básicas y no cubren todos los escenarios posibles de entrada anómala.
- La evaluación actual no incluye comparativas con otros modelos similares.
- El rendimiento puede variar dependiendo del LLM utilizado.
