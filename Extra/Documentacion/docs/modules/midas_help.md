# Midas Help

## Descripción General
MIDAS Help constituye el componente de asistencia y documentación interactiva del sistema MIDAS, más a nivel de implementación. Se trata de un chatbot inteligente basado en una arquitectura LLM+RAG+Reranker que permite a los usuarios resolver dudas sobre la implementación del sistema MIDAS mediante lenguaje natural. 

Esta arquitectura utiliza una aproximación RAG mejorada, gracias a incorporar un reranker y un selector de LLM inteligente, pero sin llegar a características avanzadas como "Agentic RAG" o bases de datos vectoriales. Todo el flujo está basado en el framework Llama-Index.

## Arquitectura Técnica

### Backend
El backend está desarrollado en Python utilizando el framework Flask y se encarga de procesas las consultas de los usuarios. Los componentes principales son:

- **Clasificador de Preguntas (Fine-tuned BERT):** Un modelo BERT afinado que *analiza la pregunta del usuario (prompt)* y la clasifica en una de tres categorías:
    -   **Pregunta fácil:** Requiere una respuesta sencilla y directa.
    -   **Pregunta difícil:** Implica una respuesta más compleja y elaborada.
    -   **Pregunta no relacionada:** No tiene relación con la documentación de MIDAS. *En este caso, el sistema no genera una respuesta.*
- Framework **Llama Index** para la generación y gestión del índice documental.
- Modelo de **embeddings BGE-M3** de BAAI para la representación vectorial de los textos (tanto de la consulta como de los documentos). Para cada consulta, se seleccionan los 30 chunks mas relevantes según su similitud vectorial.
- **Reranker BGE V2 M3:** Este componente reordena los resultados obtenidos por la búsqueda inicial basada en embeddings.  El reranker evalúa la relevancia de cada documento recuperado *con respecto a la consulta específica del usuario*, utilizando un modelo de lenguaje más sofisticado que la simple comparación de embeddings. Esto ayuda a filtrar el ruido y a asegurar que los documentos más relevantes sean presentados al LLM para la generación de la respuesta final. Toma los 30 chunks que salen del proceso de embedding, y los "filtra" para pasarle al LLM solo los 10 realmente mas relevantes.
- **Selector de LLM:** Permite elegir entre diferentes modelos de lenguaje, o usar el modo automatico para usar un modelo u otro dependiendo de la clasificación del BERT Fine-tuneado:
    -   **Modo Automático:** Utiliza el clasificador de preguntas (BERT) para seleccionar el LLM óptimo (Llama o Gemini).
    -   **Llama 3.3 70B:** Un modelo de lenguaje eficiente, ideal para preguntas fáciles.  *(Usado por defecto en el modo automático si la pregunta se clasifica como "fácil").*
    -   **Gemini 2.0 Flash:** Un modelo más potente, diseñado para preguntas difíciles que requieren mayor capacidad de razonamiento. *(Usado por defecto en el modo automático si la pregunta se clasifica como "difícil").*

### Frontend
La interfaz de usuario está construida con HTML, JavaScript y Tailwind CSS, proporcionando una experiencia moderna y responsive.

## Funcionalidad
MIDAS Help facilita:

- Acceso interactivo a la documentación técnica del sistema
- Resolución de consultas sobre implementación y arquitectura
- Comprensión de la integración entre componentes
- Soporte tanto a desarrolladores como usuarios finales

## Guía de Uso
El sistema es accesible a través de [help.midastfm.com](https://help.midastfm.com). Los usuarios pueden realizar consultas como:

- "¿Qué componentes integran MIDAS?"
- "¿Qué tipo de gráficos soporta MIDAS Plot?"
- "¿Cuál es el flujo de interacción entre componentes en MIDAS Hub?"
- "¿Qué framework utiliza MIDAS Deploy para generar interfaces Streamlit?"

Las respuestas se presentan y renderizan en formato Markdown para optimizar la legibilidad.
Mientras el sistema procesa la consulta, se muestra información en tiempo real sobre la etapa actual del proceso (por ejemplo, "Clasificando pregunta...", "Extrayendo embeddings...", "Aplicando reranking...", "Redactando respuesta..."). Se visualiza en todo momento qué LLM fue usado para la respuesta, ya sea si lo escogió automáticamente o si el usuario forzó su uso a través del selector.

## Referencias y Recursos

- Aplicación: [help.midastfm.com](https://help.midastfm.com)
- Repositorio: [github.com/warc0s/MIDAS](https://github.com/warc0s/MIDAS)
- Sitio Web Llama Index: [llamaindex.ai](https://www.llamaindex.ai)

## Limitaciones Actuales

La implementación actual no incluye:

- Sistema de RAG Agéntico
- Bases de datos vectoriales para optimización de la velocidad de búsqueda

La expansión de estas capacidades fue contemplada, pero no implementada por falta de tiempo.

## Capturas:

![Midas Imagen](https://github.com/warc0s/MIDAS/blob/main/Extra/Imagenes/Midas_Help_7_3.png?raw=true)

![Midas Imagen](https://github.com/warc0s/MIDAS/blob/main/Extra/Imagenes/Help_Full_Captura.png?raw=true)