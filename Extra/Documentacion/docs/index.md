# MIDAS: Multi-agent Intelligent Data Automation System

![MIDAS Logo](https://github.com/warc0s/MIDAS/raw/main/Extra/logo1.png)

## Transformando Ideas en Oro

MIDAS es un sistema multiagente diseñado para automatizar y optimizar el ciclo completo de ciencia de datos. Su nombre proviene de la figura mitológica del Rey Midas, cuyo toque convertía objetos en oro, simbolizando cómo este sistema transforma datos crudos en valiosos modelos predictivos y visualizaciones.

## Capacidades Principales

MIDAS ofrece un conjunto completo de herramientas para científicos de datos, desarrolladores y analistas:

- **Generación de Datos Sintéticos**: Creación automática de datasets realistas para testing y desarrollo
- **Automatización de ML**: Transformación de datos en modelos predictivos sin intervención manual
- **Evaluación de Modelos**: Análisis exhaustivo de calidad, rendimiento y robustez
- **Visualización Inteligente**: Creación de gráficos mediante descripciones en lenguaje natural
- **Despliegue Rápido**: Generación automática de interfaces para modelos
- **Asistencia y Documentación**: Sistemas avanzados de soporte basados en RAG

## Componentes del Sistema

![Midas Main Website](https://github.com/warc0s/MIDAS/blob/main/Extra/Imagenes/Midas_Main.png?raw=true)

MIDAS está compuesto por ocho módulos especializados que pueden funcionar de manera independiente o como parte de un flujo de trabajo integrado:

### [Midas Dataset](./modules/midas_dataset.md)
Generador de datasets sintéticos que utiliza agentes conversacionales basados en AG2 para interpretar requisitos y crear datos realistas mediante la biblioteca Faker.

### [Midas Touch](./modules/midas_touch.md)
Motor de automatización de ML que transforma datasets en modelos entrenados, implementando un enfoque multigente con Python vanilla y Gemini 2.0 Flash para gestionar el proceso completo.

### [Midas Test](./modules/midas_test.md)
Evaluador de modelos que analiza la calidad, rendimiento y robustez mediante agentes especializados basados en AG2, generando informes detallados en formato Markdown.

### [Midas Deploy](./modules/midas_deploy.md)
Generador de interfaces que crea aplicaciones Streamlit personalizadas para modelos ML, utilizando agentes conversacionales para analizar y diseñar la mejor experiencia de usuario.

### [Midas Plot](./modules/midas_plot.md)
Creador de visualizaciones que transforma descripciones en lenguaje natural en gráficos utilizando CrewAI Flow y ejecución segura de código en un entorno sandbox.

### [Midas Architect](./modules/midas_architect.md)
Sistema RAG agéntico que proporciona acceso inteligente a documentación técnica de frameworks como Pydantic AI, LlamaIndex, CrewAI y AG2, utilizando Supabase como base de datos vectorial.

### [Midas Help](./modules/midas_help.md)
Asistente de documentación que implementa una arquitectura LLM+RAG+Reranker para resolver consultas sobre el sistema MIDAS mediante lenguaje natural.

### [Midas Assistant](./modules/midas_assistant.md)
Chatbot inteligente que proporciona orientación, recomendaciones y soporte informativo sobre todos los componentes del ecosistema MIDAS.

## Primeros Pasos

Para comenzar a utilizar MIDAS, siga estos pasos:

1. **Instalación**.

2. **Configuración**: Configure las credenciales necesarias en los example.env

3. **Flujos de trabajo recomendados**:
   - Para crear y entrenar un modelo desde cero: Dataset → Touch → Test → Deploy
   - Para visualizar datos existentes: Plot
   - Para obtener ayuda y documentación: Assistant o Help

## Propósito y Filosofía

MIDAS nace de la visión de democratizar y automatizar los procesos de ciencia de datos mediante el uso de tecnologías de IA conversacional. El sistema busca:

1. **Reducir la barrera de entrada** para tareas complejas de ML
2. **Aumentar la productividad** de científicos de datos experimentados
3. **Mejorar la calidad** mediante evaluaciones estandarizadas
4. **Facilitar la documentación** y comprensión de procesos técnicos
5. **Promover las mejores prácticas** en el desarrollo de modelos

## Recursos Adicionales

- [Arquitectura del Sistema](./architecture.md)
- [Preguntas Frecuentes](./faq.md)
- [Repositorio GitHub](https://github.com/warc0s/MIDAS)

## Agradecimientos

MIDAS ha sido desarrollado como un Trabajo Fin de Máster (TFM) y se beneficia de múltiples frameworks y tecnologías de código abierto como AG2, CrewAI, Streamlit, Pandas, Scikit-learn y otros.

[Empezar →](/modules/midas_assistant)