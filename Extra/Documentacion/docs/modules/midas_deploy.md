# Midas Deploy

## Descripción General
MIDAS Deploy es el componente MIDAS que genera interfaces de usuario personalizadas para modelos de ML. Utilizando tecnologías de IA, específicamente LLMs, MIDAS Deploy analiza modelos guardados en formato joblib y crea aplicaciones Streamlit que permiten a los usuarios interactuar con estos modelos sin necesidad de programación adicional.

El sistema utiliza AG2 para orquestar una conversación entre agentes de IA especializados que analizan el modelo, diseñan una interfaz y generan código ejecutable.

## Arquitectura Técnica

### Backend:
- **Lenguaje y Frameworks:** 
  - *Python* como lenguaje base
  - *AG2* para la orquestación de agentes de IA
  - *Scikit-learn* para procesamiento de modelos ML
  - *Joblib* para carga y manipulación de modelos

- **Componentes clave:**
  - *Model_Analyzer*: Agente especializado que analiza modelos ML y extrae información relevante (características, parámetros, estructura)
  - *UI_Designer*: Agente encargado de diseñar la interfaz de usuario basada en el análisis del modelo
  - *Code_Generator*: Agente que implementa código funcional de Streamlit basado en el diseño de UI
  - *User_Proxy*: Orquestador del flujo de trabajo entre agentes especializados
  - *process_joblib*: Función utilitaria para extraer información de archivos joblib

- **Modelo LLM utilizado:** 
  - Meta-Llama/Llama-3.3-70B-Instruct-Turbo a través de la API de DeepInfra

- **Flujo de procesamiento:**
  1. Carga del modelo desde archivo joblib
  2. Extracción de metadatos (características, número de features, tipo de modelo)
  3. Análisis del modelo por agentes de IA
  4. Diseño de interfaz adaptada al modelo específico
  5. Generación de código Streamlit ejecutable
  6. Entrega del código para implementación

### Frontend:
- **Tecnología:** Aplicación web Streamlit
- **Componentes de UI:**
  - Cargador de archivos para modelos joblib
  - Campo de texto para descripción del modelo
  - Botón de generación de interfaz
  - Visualizador de código generado
  - Funcionalidad de descarga de código

## Funcionalidad
- Análisis automatizado de modelos de aprendizaje automático compatibles con scikit-learn
- Diseño inteligente de interfaces adaptadas a las especificaciones del modelo
- Generación de código Streamlit listo para usar
- Soporte para diversos tipos de modelos ML (clasificadores, regresores, pipelines)
- Creación de interfaces que tienen en cuenta los requisitos de entrada del modelo
- Capacidades de exportación y descarga de código
- Interacción con múltiples agentes de IA para optimizar la experiencia del usuario

## Guía de Uso
1. **Iniciar la aplicación:**
   - Ejecutar *streamlit run app.py*
   - Se abrirá la interfaz web de MIDAS Deploy en el navegador

2. **Cargar un modelo:**
   - Utilizar el cargador de archivos para subir un modelo .joblib
   - Proporcionar una breve descripción del propósito del modelo (ej. "Predicción de satisfacción del cliente basada en datos demográficos")

3. **Generar la interfaz:**
   - Hacer clic en el botón "🚀 Iniciar generación de interfaz"
   - Esperar mientras el sistema analiza el modelo y genera la interfaz

4. **Implementar el resultado:**
   - Descargar el código generado mediante el botón "📥 Descargar código generado"
   - Guardar el código como archivo .py
   - Ejecutar *streamlit run generated_interface.py*
   - La interfaz personalizada para el modelo estará disponible a través del navegador

**Ejemplo práctico:**
Para un modelo que predice la probabilidad de una condición médica basada en edad, altura y peso:
- Cargar el archivo model.joblib
- Describir como "Modelo de predicción de condición médica basado en factores biométricos"
- MIDAS Deploy generará una aplicación Streamlit con campos de entrada para edad, altura y peso
- La aplicación permitirá a los usuarios ingresar estos datos y obtener predicciones en tiempo real

## Implementación Técnica
MIDAS Deploy utiliza ConversableAgent de AG2 para crear agentes especializados:

1. **Model_Analyzer**: Analiza el modelo joblib y extrae metadatos como:
   - Tipo de modelo
   - Número de características
   - Nombres de características (si están disponibles)
   - Parámetros del modelo
   - Estructura del pipeline (si aplica)

2. **UI_Designer**: Diseña una interfaz adaptada al modelo basándose en:
   - El número de características requeridas
   - La descripción del propósito del modelo
   - El tipo de predicción (clasificación o regresión)

3. **Code_Generator**: Crea código Streamlit funcional que:
   - Carga correctamente el modelo joblib
   - Implementa campos de entrada para todas las características necesarias
   - Procesa adecuadamente los datos de entrada
   - Muestra el resultado de la predicción del modelo
   
4. **User_Proxy**: Orquesta la conversación entre los agentes, siguiendo un flujo secuencial de análisis, diseño y generación.

## Referencias y Recursos
- Documentación de AG2: https://docs.ag2.ai/docs/home/home
- Documentación de Streamlit: https://docs.streamlit.io/
- DeepInfra (para acceso a LLM): https://deepinfra.com/
- Scikit-learn (para modelos ML): https://scikit-learn.org/

## Limitaciones Actuales
- Solo soporta modelos compatibles con scikit-learn guardados en formato joblib
- Opciones de personalización limitadas para la interfaz generada
- Puede generar interfaces que necesiten ajustes menores para modelos complejos

## Capturas:

![Midas Imagen](https://github.com/warc0s/MIDAS/blob/main/Extra/Imagenes/Midas_Deploy_8_2.png?raw=true)