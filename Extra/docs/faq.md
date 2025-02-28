# Preguntas Frecuentes (FAQ)

## Preguntas Generales

### ¿Qué es MIDAS?
MIDAS (Multi-agent Intelligent Data Automation System) es un sistema multiagente diseñado para automatizar y optimizar el ciclo completo de ciencia de datos, desde la generación de datasets hasta el despliegue de modelos, utilizando tecnologías de IA conversacional y LLMs.

### ¿Por qué se llama MIDAS?
El nombre hace referencia al Rey Midas de la mitología griega, cuyo toque convertía objetos en oro. De manera similar, este sistema transforma datos crudos (datasets CSV) en "oro" (modelos de ML bien entrenados y precisos).

### ¿Cuáles son los componentes principales de MIDAS?
MIDAS consta de ocho componentes principales:
- Midas Dataset: Generador de datasets sintéticos
- Midas Touch: Automatización de flujo completo de ML
- Midas Test: Evaluador de calidad de modelos
- Midas Deploy: Generador de interfaces para modelos
- Midas Plot: Creador de visualizaciones desde lenguaje natural
- Midas Architect: Sistema RAG para documentación técnica
- Midas Help: Asistente de documentación con RAG+Reranker
- Midas Assistant: Chatbot de orientación sobre el sistema

### ¿MIDAS es un único programa o varios independientes?
MIDAS es un sistema compuesto por múltiples componentes independientes que pueden funcionar de forma autónoma o como parte de un flujo de trabajo integrado. Cada componente está diseñado para resolver una parte específica del proceso de ciencia de datos.

### ¿Qué tecnologías utiliza MIDAS?
MIDAS utiliza diversas tecnologías, incluyendo:
- Frameworks de agentes: AG2 (fork mejorado de AutoGen), CrewAI, Pydantic AI
- Modelos de lenguaje: Llama 3.3, Gemini 2.0, Deepseek V3...
- Interfaces: Streamlit, Flask
- Procesamiento de datos: Pandas, Scikit-learn
- Visualización: Matplotlib
- Bases de datos: Supabase
- Otros: LiteLLM, Faker, e2b Sandbox...

## Uso y Funcionalidad

### ¿Cómo empiezo a usar MIDAS?
Para comenzar, debe instalar los componentes que desee utilizar y configurar las credenciales necesarias para acceder a los servicios de LLM. Luego puede ejecutar cada componente individualmente según sus necesidades.

### ¿Necesito conocimientos de programación para usar MIDAS?
Los componentes de MIDAS están diseñados con interfaces intuitivas que reducen la necesidad de programación. Sin embargo, cierto conocimiento básico de ciencia de datos y ML ayudará a comprender mejor los resultados y a formular prompts efectivos.

### ¿Qué tipos de modelos de ML puede crear MIDAS?
Actualmente, Midas Touch se centra en modelos de clasificación y regresión utilizando algoritmos de Scikit-learn, específicamente RandomForest y GradientBoosting.

### ¿Qué formatos de datos acepta MIDAS?
MIDAS puede trabajar con diversos formatos:
- Midas Touch: CSV, Excel, Parquet, JSON
- Midas Plot: CSV
- Midas Test/Deploy: Modelos en formato joblib

### ¿Puedo integrar MIDAS con mis flujos de trabajo existentes?
Sí, los componentes de MIDAS están diseñados para ser modulares. Puede utilizar Midas Dataset para generar datos, procesar estos datos con sus propias herramientas, y luego usar Midas Test para evaluar los modelos resultantes.

## Capacidades y Limitaciones

### ¿Qué tamaño de datasets puede manejar MIDAS?
Midas Touch está optimizado para datasets de tamaño pequeño a mediano (recomendable hasta ~25K filas). Datasets muy grandes pueden causar problemas de rendimiento.

### ¿MIDAS requiere conexión a internet?
Sí, la mayoría de los componentes dependen de servicios externos de LLM como DeepInfra o Google AI, por lo que requieren conexión a internet para funcionar.

### ¿Qué credenciales API necesito para usar MIDAS?
Dependiendo de los componentes que utilice, puede necesitar:
- API key de DeepInfra (para componentes que usan Llama 3.3)
- API key de Google AI (para componentes que usan Gemini)

### ¿MIDAS puede explicar sus decisiones?
Sí, un enfoque clave de MIDAS es la explicabilidad. Midas Touch genera notebooks detallados que documentan cada paso del proceso, Midas Test proporciona informes completos, y Midas Deploy incluye comentarios en el código generado.

### ¿Cuáles son las limitaciones actuales más importantes?
Algunas limitaciones importantes incluyen:
- Soporte limitado de modelos ML (principalmente Scikit-learn)
- Optimización para datasets de tamaño pequeño a mediano
- Ausencia de optimización avanzada de hiperparámetros
- Falta de integración completa entre todos los componentes
- Dependencia de servicios externos para LLMs

## Problemas Comunes

### El LLM no responde o da errores de timeout
Asegúrese de que sus credenciales API estén correctamente configuradas y que tenga una conexión estable a internet. Los servicios de LLM pueden tener límites de velocidad o períodos de mantenimiento que afecten la disponibilidad.

### El modelo generado no tiene buena precisión
La calidad del modelo depende en gran medida de los datos de entrada. Asegúrese de que su dataset tenga suficientes ejemplos, características relevantes y esté correctamente preparado. Puede probar con diferentes prompts en Midas Touch para especificar mejor el objetivo.

### Midas Plot no genera la visualización que esperaba
Las descripciones en lenguaje natural pueden ser interpretadas de diferentes maneras. Intente ser más específico en su prompt, mencionando el tipo exacto de gráfico, las variables a utilizar y cualquier personalización deseada.

### Los agentes parecen "atascarse" en una conversación infinita
En raras ocasiones, los sistemas multiagente pueden entrar en bucles de conversación. Si observa que un componente no avanza después de varios minutos, puede intentar reiniciar el proceso con un prompt más claro o directivas más específicas.

## Desarrollo y Contribución

### ¿MIDAS es de código abierto?
Sí, MIDAS es un proyecto de código abierto desarrollado como Trabajo Fin de Máster (TFM). Puede encontrar el código fuente en [GitHub](https://github.com/warc0s/MIDAS).

### ¿Cómo puedo contribuir al proyecto?
Las contribuciones son bienvenidas. Puede contribuir reportando problemas, sugiriendo mejoras o enviando pull requests al repositorio GitHub.