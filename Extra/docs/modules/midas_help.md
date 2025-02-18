# Componente MIDAS Help

## Descripción General
MIDAS Help constituye el componente de asistencia y documentación interactiva del sistema MIDAS. Implementa un chatbot inteligente basado en una arquitectura LLM+RAG que permite a los usuarios resolver dudas sobre el funcionamiento del sistema mediante lenguaje natural. Esta implementación utiliza una aproximación básica de RAG, sin incorporar características avanzadas como "Agentic RAG" o bases de datos vectoriales, y está basado en el framework Llama Index.

## Arquitectura Técnica

### Backend
El backend está desarrollado en Python utilizando el framework Flask y se encarga del procesamiento de consultas mediante técnicas RAG. Los componentes principales son:

- Framework **Llama Index** para la generación y gestión del índice documental.
- Modelo de **embeddings BGE-M3** de BAAI para comparación semántica.
- Modelo de lenguaje **Llama 3.3 70b** para generación de respuestas.

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

## Referencias y Recursos

- Aplicación: [help.midastfm.com](https://help.midastfm.com)
- Repositorio: [github.com/warc0s/MIDAS](https://github.com/warc0s/MIDAS)
- Sitio Web Llama Index: [llamaindex.ai](https://www.llamaindex.ai)

## Limitaciones Actuales

La implementación actual no incluye:

- Sistema de RAG Agéntico
- Bases de datos vectoriales para optimización de la velocidad de búsqueda

La expansión de estas capacidades fue contemplada, pero no implementada por falta de tiempo.
