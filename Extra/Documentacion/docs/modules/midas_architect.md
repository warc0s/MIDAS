# Midas Architect

## Descripción General

Midas Architect es un componente de Recuperación Aumentada Generativa (RAG) que utiliza Supabase como base de datos vectorial para almacenar y consultar documentación técnica de cuatro frameworks de desarrollo: Pydantic AI, LlamaIndex, CrewAI y AG2.

Este sistema implementa un enfoque de RAG asistido por agentes, permitiendo navegar inteligentemente por la documentación técnica mediante el uso de herramientas específicas de consulta. Utiliza modelos de lenguaje grandes (LLM), específicamente Gemini 2.0 Flash, para procesar consultas y generar respuestas contextualizadas basadas en la documentación oficial de estos frameworks.

## Arquitectura Técnica

### Backend:

- **Sistema de Ingesta de Documentación**:
  - Utiliza *Crawl4AI* para extraer automáticamente contenido en formato Markdown de los sitemaps oficiales de cada framework.
  - Procesa cada página web recuperada y la convierte a un formato optimizado para su posterior procesamiento.

- **Procesamiento de Texto**:
  - Implementa una *segmentación inteligente* que divide el texto en chunks de máximo 5000 caracteres.
  - La segmentación respeta las siguientes estructuras para mantener la coherencia contextual:
    - *Bloques de código*: Detecta marcadores "```" después del 30% del chunk.
    - *Párrafos*: Identifica saltos de línea dobles "\n\n" después del 30% del chunk.
    - *Oraciones*: Localiza finales de oración ". " después del 30% del chunk.
  - Esta estrategia garantiza chunks de tamaño óptimo para el procesamiento por LLMs.

- **Sistema de Embeddings**:
  - Utiliza el modelo *text-embedding-3-small* de OpenAI (1536 dimensiones) para generar representaciones vectoriales del texto.
  - Implementa el modelo *gpt-4o-mini* para la generación automática de títulos y resúmenes de cada chunk.

- **Base de Datos Vectorial**:
  - *Supabase* como infraestructura para almacenar embeddings y metadatos.
  - Estructura de tabla SQL optimizada para consultas vectoriales mediante índices IVFFlat.
  - Cada registro incluye: *embedding vectorial*, *URL de origen*, *título*, *resumen*, *contenido completo* y *metadatos* (incluyendo la fuente del documento).

- **Sistema de Consulta Basado en Herramientas**:
  - Implementa tres herramientas principales mediante Pydantic AI:
    - *retrieve_relevant_documentation*: Recuperación basada en similitud de embeddings.
    - *list_documentation_pages*: Listado de todas las URLs disponibles para un framework específico.
    - *get_page_content*: Recuperación de todos los chunks de una página específica mediante URL exacta.

### Frontend:
- Implementado en Streamlit con diseño responsivo y experiencia de usuario mejorada.
- Interfaz con estilos personalizados y animaciones para una mejor experiencia.
- Selector de framework que permite cambiar dinámicamente entre las diferentes fuentes de documentación.
- Sistema de streaming de respuestas en tiempo real.

## Funcionalidad

- Proporciona respuestas precisas a consultas técnicas sobre los frameworks Pydantic AI, LlamaIndex, CrewAI y AG2.
- Ofrece capacidad de comprensión y contextualización profunda de la documentación técnica.
- Permite la recuperación selectiva e inteligente de información relevante mediante enfoque agéntico.
- Facilita el acceso a información técnica compleja sin necesidad de navegar manualmente por la documentación.
- Responde en español a pesar de que la documentación original está en inglés.
- Dirigido principalmente a desarrolladores que trabajan con estos frameworks y buscan resolver dudas técnicas de forma rápida.

## Guía de Uso

Para interactuar con Midas Architect:

1. **Seleccionar el framework** sobre el que se desea consultar información mediante el selector en la barra lateral.

2. **Formular consultas específicas** en español sobre el framework seleccionado.
  
   *Ejemplo de consulta:* "¿Cómo puedo implementar un RAG básico con LlamaIndex?"

3. El sistema procesará la consulta a través de su pipeline:
   - Analizará la consulta para entender qué información se necesita.
   - Recuperará chunks relevantes de la documentación mediante similitud vectorial.
   - Si es necesario, consultará páginas completas o listará recursos disponibles.
   - Generará una respuesta detallada en español basada en la documentación original.

## Referencias y Recursos

- Modelo de embeddings: [OpenAI text-embedding-3-small](https://platform.openai.com/docs/guides/embeddings)
- Modelo para resúmenes y títulos: OpenAI gpt-4o-mini
- Modelo principal de LLM: Gemini 2.0 Flash
- Base de datos vectorial: [Supabase Vector](https://supabase.com/docs/guides/ai)
- Frameworks documentados:
  - [Pydantic AI](https://docs.pydantic.dev/)
  - [LlamaIndex](https://docs.llamaindex.ai/)
  - [CrewAI](https://docs.crewai.com/)
  - [AG2](https://docs.ag2.ai/docs/user-guide/basic-concepts/installing-ag2)
- Librería de crawling: [Crawl4AI](https://github.com/unclecode/crawl4ai)

## Limitaciones Actuales

- La documentación de LlamaIndex está incompleta debido a su extensión (más de 1650 páginas), lo que puede afectar a la capacidad del sistema para responder algunas consultas específicas sobre este framework.
- No se ha implementado un sistema de citas de fuentes para las respuestas. Los intentos de incluir fuentes mediante prompting resultaron en la generación de URLs inexistentes (alucinadas).
- El modelo Gemini 2.0 Flash puede tener limitaciones en el procesamiento de consultas muy específicas o complejas.
- Sistema diseñado para consultas en español únicamente a pesar de que la documentación original está en inglés.

## Capturas:

![Midas Imagen](https://github.com/warc0s/MIDAS/blob/main/Extra/Imagenes/Midas_Architech.png?raw=true)