# Midas Assistant

## Descripción General
MIDAS Assistant es el componente del sistema MIDAS que proporciona orientación, recomendaciones y soporte informativo sobre todos los componentes del ecosistema MIDAS. Actúa como un chatbot inteligente utilizando tecnología LLM para responder consultas relacionadas con el sistema MIDAS, sus componentes individuales y flujos de trabajo óptimos.

Este componente se basa en LiteLLM como framework de abstracción, permitiendo la integración con diferentes modelos de lenguaje como Gemini, dependiendo de la configuración del usuario. Básicamente, a grandes rasgos, es un LLM con un gran system prompt con información acerca de cada componente Midas para así resolver dudas sobre el mismo.

## Arquitectura Técnica

### Backend
- **Lenguaje y framework:** 
 - *Python* como lenguaje principal
 - *LiteLLM* como framework de abstracción para interactuar con LLMs
 - *Flask* para la versión web

- **Componentes clave:**
 - *Módulo de configuración:* Gestiona las variables de entorno y la configuración del modelo a utilizar
 - *Gestor de contexto:* Mantiene el historial de conversación para proporcionar respuestas contextualizadas
 - *Sistema de prompts:* Utiliza un prompt de sistema extenso con información detallada sobre todos los componentes MIDAS
 - *API REST:* En la versión Flask, proporciona endpoints para consultas y gestión de conversaciones

- **Flujo de procesamiento:**
 1. Recepción de la consulta del usuario
 2. Consulta al LLM configurado vía LiteLLM
 3. Formateo y entrega de la respuesta al usuario

### Frontend
- **Versión CLI:**
 - Terminal interactiva con *Colorama* para destacar elementos visuales
 - Formato de texto para mejorar la legibilidad de las respuestas

- **Versión Web:**
 - *HTML/CSS* con *Tailwind CSS* para una interfaz moderna y responsiva
 - *JavaScript* para la gestión del chat y efectos visuales
 - *Marked.js* para renderizar Markdown de las respuestas del LLM

## Funcionalidad
- Proporciona información completa sobre los ocho componentes del sistema MIDAS
- Genera recomendaciones de flujos de trabajo adaptados a las necesidades del usuario
- Sugiere prompts efectivos para interactuar con cada componente específico
- Direcciona consultas técnicas específicas hacia Midas Help - Dado que la idea es sugerir usos de los componentes Midas, no responder dudas sobre el TFM.
- Mantiene un tono profesional y conciso, enfocado en proporcionar valor práctico
- Presenta la información en formato Markdown para una mejor legibilidad

## Guía de Uso

### Versión CLI
1. Configura tus credenciales en el archivo `.env` (siguiendo el formato de `example.env`)
2. Ejecuta el script `Midas_Assistant_cli.py`
3. Inicia el diálogo con preguntas como:
  - "¿Qué componente MIDAS debo usar para visualizar datos?"
  - "Dame un prompt efectivo para Midas Plot"
  - "¿Cómo implemento un flujo de trabajo para crear un modelo predictivo?"

### Versión Web
1. Configura tus credenciales en el archivo `.env`
2. Ejecuta `Midas_Assitant_flask.py` para iniciar el servidor
3. Accede a la interfaz web desde tu navegador
4. Interactúa con el chatbot mediante el campo de texto
5. Utiliza el panel de componentes para acceder rápidamente a información específica

**Ejemplo de interacción:**
- Usuario: "Necesito crear un dataset y visualizarlo para analizar tendencias"
- MIDAS Assistant: "Para ese flujo de trabajo te recomiendo usar MIDAS DATASET para generar tus datos sintéticos, especificando el número de filas y columnas necesario. Luego, utiliza MIDAS PLOT para visualizar las tendencias. Para MIDAS PLOT, un prompt efectivo sería: 'Genera una gráfica de líneas temporal que muestre la evolución de [variable] agrupada por [categoría]'."

## Referencias y Recursos
- Repositorio GitHub: [MIDAS](https://github.com/warc0s/MIDAS)
- Website de LiteLLM: [LiteLLM Documentation](https://litellm.ai/)

## Limitaciones Actuales
- El componente está optimizado para responder sobre el ecosistema MIDAS, rechazando educadamente consultas fuera de este ámbito
- La calidad de respuesta depende del modelo LLM configurado, siendo gemini-2.0-flash el mejor calidad/precio de todos los que hemos probado
- La versión CLI no conserva el historial de conversación entre sesiones (aunque la versión web sí lo hace)
- No existe integración directa con otros componentes MIDAS, es puramente informativo
- La idea original era implementarlo como un agente que tuviera como herramientas cada componente MIDAS, de forma que con un prompt simple como "hazme un modelo ML que prediga X" fuera capaz de invocar automáticamente estas herramientas con los mejores prompts posibles que el agente conoce y devolviera exactamente lo que el usuario necesita. Sin embargo, debido a limitaciones de tiempo, esta funcionalidad no pudo ser implementada.

## Capturas:

![Midas Imagen](https://github.com/warc0s/MIDAS/blob/main/Extra/Imagenes/Midas_Assistant.png?raw=true)