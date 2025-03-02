# Midas Test

## Descripción General

MIDAS Test es el componente MIDAS especializado en la evaluación exhaustiva de modelos de machine learning almacenados en formato joblib. Su propósito principal es analizar la calidad, rendimiento y robustez de modelos ML mediante una arquitectura de agentes conversacionales basados en IA.

El sistema utiliza Large Language Models (LLM) para coordinar múltiples agentes especializados que evalúan diferentes aspectos de los modelos ML. MIDAS Test implementa un enfoque de colaboración multi-agente donde cada agente aporta su perspectiva especializada para generar un informe completo.

MIDAS Test se basa en el framework AG2 para la gestión de agentes conversacionales y utiliza Streamlit para proporcionar una interfaz de usuario accesible.

## Arquitectura Técnica

### Backend:

- **Lenguaje y Bibliotecas:** 
  - Python 3.x
  - AG2 para la gestión de agentes IA
  - Scikit-learn para manipulación de modelos ML
  - Joblib para carga/guardado de modelos
  - DeepInfra API para acceder a modelos LLM
  - deep_translator para traducir informes al español

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
    - **check_model_validity**: Verifica si el modelo es compatible con Scikit-learn.
    - **measure_latency**: Evalúa tiempos de respuesta en diferentes tamaños de batch.
    - **measure_memory_usage**: Mide el uso de memoria.
    - **measure_memory_and_cpu_during_prediction**: Evalúa el uso de recursos durante predicciones.
    - **validate_predictions**: Verifica la consistencia y formato de las predicciones.
    - **check_robustness**: Prueba comportamiento ante valores nulos, extremos y tipos incorrectos.
    - **translate_to_spanish**: Traduce el informe al español.
    - **generate_markdown_report**: Compila los hallazgos en formato Markdown estructurado.

- **Flujo de Procesamiento**:
  1. Carga del modelo joblib.
  2. Validación inicial del modelo (compatibilidad con Scikit-learn).
  3. Generación de datos de muestra para pruebas.
  4. Ejecución de pruebas de rendimiento, robustez y validación.
  5. Compilación de métricas y resultados.
  6. Activación de agentes IA para análisis especializado.
  7. Generación de informe final en formato Markdown en español.

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

- **Análisis de Modelos ML**: Evalúa múltiples aspectos del modelo incluyendo validez, rendimiento y robustez.

- **Métricas de Rendimiento**: 
  - Tiempo de carga del modelo
  - Uso de memoria durante predicciones
  - Utilización de CPU
  - Latencia en diferentes tamaños de batch (1, 100, 1000, 10000)
  - Throughput (predicciones por segundo)

- **Pruebas de Robustez**:
  - Comportamiento ante valores nulos
  - Resistencia a valores fuera de rango
  - Manejo de tipos de datos incorrectos
  - Consistencia de predicciones

- **Validación de Salidas**:
  - Verificación de formato correcto (array NumPy)
  - Validación de rangos de valores
  - Comprobación de suma de probabilidades igual a 1 (cuando aplica)

- **Recomendación Automatizada**: Clasificación del modelo como "APTO" o "NO APTO" basada en la validez del modelo y la consistencia de sus predicciones.

- **Reporte Markdown**: Generación automática de documentación estructurada en español con los hallazgos y recomendaciones.

## Guía de Uso

### A través de la Interfaz Web (Streamlit):

1. Inicie la aplicación ejecutando:
   *streamlit run app.py*

2. En la interfaz web, haga clic en el cargador de archivos y seleccione el modelo joblib a evaluar.

3. Una vez cargado el modelo, pulse el botón "🔄 Iniciar Evaluación con los Agentes" para comenzar el análisis.

4. El sistema mostrará un mensaje indicando que la evaluación está en proceso.

5. Después de unos 90 segundos, pulse "📄 Finalizar Análisis y Descargar Reporte" para ver y descargar los resultados.

6. Explore los resultados en las secciones expandibles:
   - "📌 Información del Modelo": Datos básicos como tiempo de carga y tamaño
   - "📈 Métricas de Rendimiento": Detalles sobre uso de recursos
   - "⚠️ Pruebas de Robustez": Resultados de las pruebas de resistencia

7. Descargue el informe completo en formato Markdown utilizando el botón "⬇️ Descargar Reporte".

### Mediante Línea de Comandos:

1. Ejecute el script principal:
   *python agents_test.py*

2. Cuando se solicite, ingrese la ruta completa al archivo joblib que desea analizar.

3. El sistema ejecutará automáticamente todas las pruebas y generará un informe en el archivo "informe_analisis_modelo.md".

### Ejemplo de Salida:

El reporte generado contendrá secciones como:

# 📊 Informe de Análisis del Modelo
**Generado el:** 2025-03-02 15:30:45

---

## 🔍 Resumen del Modelo
[Información general sobre el modelo y sus características]

## ⚙️ Métricas de Rendimiento
[Detalles sobre rendimiento, memoria y CPU]

## ⏳ Análisis de Latencia
[Análisis de tiempos de respuesta]

## ✅ Validez de Predicciones
[Validación de las salidas del modelo]

## 🛡️ Pruebas de Robustez
[Resultados de pruebas de resistencia]

## 📌 Recomendación Final
**APTO**

## 🔧 Sugerencias de Mejora
[Recomendaciones para mejorar el modelo]

## Limitaciones Actuales

- El componente está optimizado para modelos Scikit-learn y puede tener limitaciones con otros frameworks de ML.
- Las pruebas de robustez son básicas y no cubren todos los escenarios posibles de entrada anómala.
- La evaluación actual se centra en la validez del modelo y consistencia de predicciones, sin métricas específicas de calidad predictiva.
- El rendimiento de los agentes puede variar dependiendo de la calidad de las respuestas del LLM utilizado.
- La traducción automática al español puede contener imprecisiones en terminología técnica.