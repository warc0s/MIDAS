# Arquitectura del Sistema MIDAS

## Visión General

MIDAS (Multi-agent Intelligent Data Automation System) es una plataforma multiagente diseñada para automatizar y optimizar el ciclo completo de ciencia de datos, desde la generación de datasets hasta el despliegue de modelos. El nombre MIDAS hace referencia al Rey Midas, cuyo toque convertía objetos en oro, simbolizando cómo este sistema transforma datos crudos (CSV) en valiosos modelos predictivos (joblib).

La arquitectura de MIDAS implementa un enfoque modular y desacoplado, donde cada componente especializado se comunica a través de interfaces bien definidas y formatos estándar. El sistema aprovecha múltiples frameworks de IA conversacional (AG2, CrewAI, LiteLLM) y modelos de lenguaje de gran escala (LLMs) para proporcionar capacidades avanzadas de automatización, razonamiento y generación.

## Componentes Principales

MIDAS está compuesto por ocho módulos especializados que pueden funcionar de manera independiente o como parte de un flujo de trabajo integrado:

1. **Midas Dataset**: Generador de datasets sintéticos basado en agentes AG2
2. **Midas Touch**: Motor de procesamiento automático de ML que transforma datos en modelos
3. **Midas Test**: Evaluador de calidad y rendimiento de modelos ML
4. **Midas Deploy**: Generador de interfaces para modelos entrenados
5. **Midas Plot**: Creador de visualizaciones mediante instrucciones en lenguaje natural
6. **Midas Architect**: Sistema RAG agéntico para documentación técnica
7. **Midas Help**: Asistente de documentación con RAG mejorado y reranking
8. **Midas Assistant**: Chatbot inteligente para navegación y orientación

## Diagrama de Arquitectura Conceptual

La arquitectura de MIDAS sigue un patrón de flujo de trabajo lineal con múltiples puntos de entrada y retroalimentación:

[MIDAS DATASET] ────┐
│
▼
[MIDAS TOUCH] ───┐
│          │
▼          ▼
[MIDAS TEST] [MIDAS PLOT]
│          │
▼          │
[MIDAS DEPLOY]◄──┘
[MIDAS ARCHITECT]
[MIDAS HELP]     } ── Sistemas de soporte transversales
[MIDAS ASSISTANT]

## Tecnologías y Frameworks

MIDAS integra múltiples tecnologías de vanguardia:

### Frameworks de IA Multi-agente:
- **AG2**: Utilizado en Midas Dataset, Midas Deploy y Midas Test para orquestar conversaciones entre agentes especializados
- **CrewAI**: Implementado en Midas Plot para gestionar flujos de trabajo de generación visual
- **Python "vanilla"**: Sistema de agentes personalizado en Midas Touch

### Modelos de Lenguaje (LLMs):
- **Meta Llama 3.3 (70B)**: Utilizado principalmente en Midas Dataset, Midas Deploy y Midas Test
- **Gemini 2.0 Flash**: Implementado en Midas Touch y como opción en Midas Help
- **Deepseek V3**: Utilizado en ciertos casos de Midas Help
- **OpenAI 4o-mini**: Para generación de resúmenes en Midas Architect

### Bases de Datos y Almacenamiento:
- **Supabase**: Como base de datos vectorial en Midas Architect
- **Sistemas de archivos locales**: Para almacenamiento de modelos y datasets

### Interfaces de Usuario:
- **Streamlit**: Implementado en todos los componentes con interfaz gráfica
- **Flask**: Utilizado en versiones web de Midas Assistant y Midas Help

### Procesamiento de Datos y ML:
- **Pandas**: Para manipulación y análisis de datos
- **Scikit-learn**: Para creación y evaluación de modelos
- **Matplotlib**: Para generación de visualizaciones

### Otros Componentes:
- **Faker**: Para generación de datos sintéticos
- **LiteLLM**: Como abstracción para interacción con diferentes LLMs
- **e2b Sandbox**: Para ejecución segura de código en Midas Plot
- **Embeddings**: Diversos modelos como text-embedding-3-small y BGE-M3

## Flujos de Datos y Comunicación

MIDAS implementa varios flujos de trabajo principales:

1. **Flujo de Generación de Modelos**:
   - Midas Dataset → Midas Touch → Midas Test → Midas Deploy
   
2. **Flujo de Visualización**:
   - Midas Dataset/Datos existentes → Midas Plot
   
3. **Flujos de Soporte**:
   - Usuario → Midas Help/Architect/Assistant → Usuario

Cada componente produce artefactos específicos que pueden servir como entradas para otros componentes:

- **Midas Dataset**: Produce archivos CSV con datos sintéticos
- **Midas Touch**: Genera modelos ML en formato joblib
- **Midas Test**: Crea informes de evaluación en Markdown
- **Midas Deploy**: Produce aplicaciones Streamlit ejecutables
- **Midas Plot**: Genera visualizaciones en formato PNG

## Consideraciones de Diseño

La arquitectura de MIDAS se basa en varios principios clave:

1. **Modularidad**: Cada componente está diseñado para funcionar de forma independiente
2. **Especialización**: Los componentes se centran en resolver tareas específicas del flujo de ML
3. **Interoperabilidad**: Uso de formatos estándar (CSV, joblib) para facilitar la integración
4. **Automatización**: Minimización de intervención manual en procesos complejos
5. **Explicabilidad**: Generación automática de documentación y visualizaciones para mejorar la comprensión
6. **Extensibilidad**: Arquitectura que permite añadir nuevos componentes o mejorar los existentes

## Limitaciones de la Arquitectura Actual

La arquitectura actual presenta algunas limitaciones que podrían abordarse en versiones futuras:

1. **Integración parcial**: Aunque conceptualmente forman un sistema, los componentes no están completamente integrados en una plataforma unificada
2. **Diversidad de frameworks**: El uso de diferentes frameworks (AG2, CrewAI) puede complicar el mantenimiento
3. **Dependencia de servicios externos**: Varios componentes dependen de APIs externas para acceder a LLMs
4. **Ausencia de orquestación central**: No existe un componente que coordine automáticamente el flujo completo
5. **Limitaciones de escalabilidad**: Algunos componentes están optimizados para datasets de tamaño pequeño a mediano

## Evolución Futura

La arquitectura de MIDAS está diseñada para evolucionar en varias direcciones:

1. **Integración más profunda** entre componentes para facilitar flujos de trabajo end-to-end
2. **Unificación de frameworks** para simplificar el mantenimiento
3. **Implementación de un orquestador central** que permita automatizar flujos de trabajo completos
4. **Extensión de capacidades** para soportar casos de uso más complejos
5. **Mejora de interfaces** para ofrecer una experiencia de usuario más coherente

[Empezar →](/modules/plot)