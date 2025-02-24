# Midas Architect

## Descripción General

Midas Architect es un componente de Recuperación Aumentada Generativa (RAG) con capacidades agénticas que utiliza Supabase como base de datos vectorial para almacenar y gestionar documentación técnica de cuatro frameworks clave: Pydantic AI, LlamaIndex, CrewAI y AG2. 

Este componente implementa un enfoque de RAG agéntico, lo que le permite razonar sobre la estructura de la documentación y tomar decisiones inteligentes sobre qué información recuperar en función de las consultas de los usuarios. Utiliza tecnologías de IA avanzadas, particularmente modelos de lenguaje grandes (LLM) para procesar y responder consultas técnicas sobre estos frameworks.

## Arquitectura Técnica

### Backend:

- **Sistema de Ingesta de Documentación**:
 - Utiliza *Crawl4AI* para extraer contenido en formato Markdown de la documentación de los frameworks objetivo.
 - El formato Markdown se selecciona específicamente por su mejor *legibilidad para LLMs*.

- **Procesamiento de Texto**:
 - *Segmentación inteligente* que divide el texto en chunks de máximo 5000 caracteres, utilizando heurísticas para encontrar los puntos de corte más adecuados.
 - La segmentación respeta las siguientes estructuras para mantener la coherencia contextual:
   - *Bloques de código*: Detecta marcadores "```" después del 30% del chunk.
   - *Párrafos*: Identifica saltos de línea dobles "\n\n" después del 30% del chunk.
   - *Oraciones*: Localiza finales de oración ". " después del 30% del chunk.
 - Esta estrategia garantiza chunks entre 375 tokens (30%) y 1250 tokens (máximo).

- **Sistema de Embeddings**:
 - Utiliza el modelo *text-embedding-3-small* de OpenAI (1536 dimensiones).
 - Implementa el modelo *4o-mini* para la generación de resúmenes de cada chunk - para ayudar al agente a razonar sobre la relevancia de dicho chunk.

- **Base de Datos Vectorial**:
 - *Supabase* como infraestructura para almacenar embeddings junto con metadatos contextuales.
 - Estructura de tabla SQL optimizada para consultas vectoriales mediante índices IVFFlat.
 - Cada registro incluye el *embedding vectorial*, *URL de origen*, *título*, *resumen del chunk*, *contenido* y *metadatos*.

- **Sistema de Consulta Agéntico**:
 - Dispone de dos herramientas (tools) principales:
   - Herramienta de recuperación basada en *similaridad de embeddings*.
   - Herramienta de razonamiento basada en *títulos y resúmenes*.
 - Capacidad para razonar sobre listas completas de URLs de documentación.
 - Funcionalidad para recuperar todos los chunks de una página específica, no solo los relevantes según similitud de embeddings.

### Frontend:
- Streamlit

## Funcionalidad

- Proporciona respuestas precisas a consultas técnicas sobre los frameworks Pydantic AI, LlamaIndex, CrewAI y AG2.
- Ofrece capacidad de comprensión y contextualización profunda de la documentación técnica.
- Permite la recuperación selectiva e inteligente de información relevante mediante enfoque agéntico.
- Facilita el acceso a información técnica compleja sin necesidad de navegar manualmente por la documentación.
- Mantiene la trazabilidad entre respuestas y fuentes documentales originales.
- Dirigido principalmente a desarrolladores que trabajan con estos frameworks y busquen resolver dudas sobre estos de forma rápida y sencilla.

## Guía de Uso

Para interactuar con Midas Architect:

1. **Formular consultas específicas** sobre cualquiera de los frameworks soportados (Pydantic AI, LlamaIndex, CrewAI o AG2).
  
  Ejemplo de consulta: "¿Cómo puedo implementar un RAG básico con LlamaIndex?"

2. El sistema procesará la consulta a través de su pipeline de RAG agéntico:
  - Analizará la consulta para identificar el framework y concepto específico.
  - Evaluará qué páginas de documentación pueden contener información relevante.
  - Recuperará chunks completos de documentación según sea necesario.
  
3. El sistema generará una respuesta detallada basada en la documentación recuperada.

## Referencias y Recursos

- Modelo de embeddings: [OpenAI text-embedding-3-small](https://platform.openai.com/docs/guides/embeddings)
- Modelo para resúmenes: OpenAI 4o-mini
- Base de datos vectorial: [Supabase Vector](https://supabase.com/docs/guides/ai)
- Frameworks documentados:
 - [Pydantic AI](https://docs.pydantic.dev/)
 - [LlamaIndex](https://docs.llamaindex.ai/)
 - [CrewAI](https://docs.crewai.com/)
 - [AG2](https://docs.ag2.ai/docs/user-guide/basic-concepts/installing-ag2)

## Limitaciones Actuales

- La documentación de LlamaIndex está incompleta debido a su extensión (más de 1650 páginas), lo que puede afectar a la capacidad del sistema para responder algunas consultas específicas sobre este framework.
- No se ha implementado un sistema de citas de fuentes para las respuestas. Los intentos de incluir fuentes mediante prompting resultaron en la generación de URLs inexistentes (alucinadas).
