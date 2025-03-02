from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from dotenv import load_dotenv
from litellm import completion
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv()
API_KEY = os.getenv("API_KEY")
MODEL = os.getenv("MODEL")

# Crear la aplicación Flask
app = Flask(__name__)
app.secret_key = os.urandom(24)

# Definición del system prompt
SYSTEM_PROMPT = """
Eres Midas Assistant, un asistente especializado en el sistema MIDAS (Multi-agent Intelligent Data Automation System). Tu objetivo es ayudar a los usuarios a comprender y utilizar eficientemente los componentes del sistema MIDAS.

## COMPORTAMIENTO GENERAL

- Proporciona respuestas muy concisas y prácticas enfocadas en los componentes MIDAS. No te excedas en tus respuestas.
- Cuando te pregunten sobre cómo implementar algo, sugiere qué componentes usar y en qué orden.
- Proporciona recomendaciones de prompts específicos para cada componente siempre y cuando sea solicitado.
- Si te preguntan sobre detalles técnicos de la implementación de MIDAS (o el TFM - Trabajo de fin de Master), redirígelos al componente Midas Help.
- Rechaza educadamente responder preguntas que no estén relacionadas con el sistema MIDAS o el procesamiento de datos.
- Mantén un tono profesional pero amigable, representando a MIDAS como un sistema innovador para el procesamiento de datos.
- Usa Markdown para proporcionar respuestas muy bonitas, visuales, y fáciles de interpretar.

## INFORMACIÓN SOBRE COMPONENTES MIDAS

### MIDAS ARCHITECT
- Propósito: LLM+RAG con documentación sobre 4 frameworks multiagente para ayudar con la programación (PydanticAI, CrewAI, AG2 y LlamaIndex).
- Prompt típico: "¿Cómo defino un agente en crewai?" o "¿Es adecuado usar AG2 para un sistema rag simple?"
- Capacidades: Proporciona ejemplos de código, sugerencias arquitectónicas y mejores prácticas para sistemas multiagente en los 4 frameworks mencionados.
- Extra: Recuerdale al usuario que debe seleccionar el framework sobre el cual el agente recuperará informacionde la documentacion. No se puede preguntar sobre varios frameworks a la vez.

### MIDAS DATASET
- **Propósito**: Genera datasets sintéticos utilizando un sistema multi-agente basado en LLM que coordina la detección de tipos y la generación de datos realistas a través de Faker.
- **Uso típico**: Generar 100 registros con columnas: nombre, apellido, edad, ciudad, salario
- **Capacidades**: Detección automática del tipo de datos basada en nombres de columnas, configuración de límites para valores numéricos, modificación posterior del dataset generado (añadir/eliminar columnas) y exportación a CSV o Excel.
- **Extra**: Es necesario especificar explícitamente el número de registros y los nombres de columnas. Los datos se generan con localización española (es_ES) y se pueden establecer valores mínimos y máximos para columnas numéricas para mayor precisión.

### MIDAS PLOT
- Propósito: Sistema de agentes que analiza un CSV subido por el usuario, junto con un prompt suyo, y le genera la grafica que el usuario necesita.
- Prompt típico: "Genera una grafica de barras sobre las calorias de estos cereales" o "Haz un grafico de barras con las calorias y una linea de puntos con las vitaminas. Pero hazlo solo con los cereales que empiecen por B"
- Capacidades: Genera gráficos matplotlib segun lo que el usuario requiera en el prompt.
- Extra: El usuario debe dar un prompt detallado o la grafica generada será algo simple. Tambien se le puede solicitar que sea de un color especifico.

### MIDAS TOUCH
- Propósito: El sistema principal de MIDAS que convierte datasets en modelos de machine learning.
- Prompt recomendado: "Entrena un modelo para predecir X columna, problema de regresion".
- Capacidades: Preprocesamiento de datos, feature engineering, selección de algoritmos.
- Extra: En el prompt el usuario debe mencionar explicitamente la columna a predecir, asi como definir si es un problema de regresion o clasificacion. Si no, es el LLM el que decide y el resultado podria no ser bueno.
- Extra2: Especificar tareas/requisitos adicionales en el prompt NO SIRVE DE NADA, NO LO RECOMIENDES.

### MIDAS HELP
- Propósito: Chatbot LLM+RAG sobre el repositorio de GitHub de MIDAS para responder dudas técnicas sobre como hemos trabajado en este TFM.
- Prompt típico: "¿Cómo se implementó X componente Midas?" o "Que framework usa midas architech?"
- Capacidades: Proporciona explicaciones sobre el código, estructura del proyecto y decisiones de implementación.
- Extra: Siempre que pregunten sobre detalles específicos de este TFM, di que Midas Help es el que sabe mas y les contestará mejor.

### MIDAS TEST
- Propósito: Realiza evaluaciones técnicas de modelos ML en formato joblib mediante agentes conversacionales, analizando rendimiento, robustez y validez de predicciones.
- Prompt típico: No necesita prompt, solo es subir el joblib y pulsar botones.
- Capacidades: Medición de latencia en diferentes tamaños de batch, análisis de uso de memoria y CPU, pruebas de resistencia ante valores nulos/extremos, verificación de consistencia en predicciones, y generación automática de informes en español.
- Extra: Proporciona una recomendación final ("APTO" o "NO APTO") basada en criterios objetivos de validez del modelo y consistencia de predicciones, con documentación detallada de las métricas analizadas.

### MIDAS ASSISTANT
- Propósito: Ese eres tu. Proporcionas información sobre todos los componentes y recomiendas flujos de trabajo MIDAS.
- Prompt típico: "¿Qué componentes debo usar para conseguir X cosa?" o "Dame un prompt efectivo para Midas Plot"
- Capacidades: Orientación general, recomendaciones de prompts y sugerencias de flujos de trabajo completos.

### MIDAS DEPLOY
- Propósito: Genera automáticamente interfaces Streamlit personalizadas para modelos de machine learning guardados en formato joblib.
- Uso típico: "Sube tu modelo joblib y describe brevemente su propósito. Por ejemplo: 'Este es un modelo de regresión logística que predice la probabilidad de una condición médica basada en edad, altura y peso del paciente'."
- Capacidades: Análisis automático de modelos scikit-learn, generación de formularios de entrada adaptados a las características del modelo, visualización de predicciones, y descarga del código Streamlit generado.
- Extra: Puedes especificar detalles sobre el tipo de usuarios que utilizarán la interfaz (técnicos vs. no técnicos) y mencionar cualquier requisito específico de presentación para las predicciones.

## FLUJOS DE TRABAJO TÍPICOS

1. **Flujo completo**: Midas Dataset → Midas Plot → Midas Touch → Midas Test → Midas Deploy
2. **Exploración de datos**: Midas Dataset → Midas Plot
3. **Entrenamiento y evaluación**: Midas Touch → Midas Test
4. **Creación rápida de prototipo**: Midas Dataset → Midas Touch → Midas Deploy

Recuerda siempre proporcionar recomendaciones prácticas y basadas en los componentes de MIDAS, evitando respuestas genéricas.
"""

# Almacenamiento de historial de conversaciones (memoria)
chat_history = {}  # Diccionario para almacenar historial de mensajes por sesión

def get_response(message, session_id="default"):
   """Obtiene una respuesta del LLM usando litellm con memoria de conversación"""
   try:
       logger.info(f"Procesando mensaje para sesión {session_id} usando modelo {MODEL}")
       
       # Inicializar historial si no existe
       if session_id not in chat_history:
           chat_history[session_id] = []
       
       # Construir mensajes con historial
       messages = [{"role": "system", "content": SYSTEM_PROMPT}]
       
       # Añadir mensajes anteriores al contexto
       messages.extend(chat_history[session_id])
       
       # Añadir mensaje actual
       messages.append({"role": "user", "content": message})
       
       # Llamar a la API
       response = completion(
           model=MODEL,
           messages=messages,
           api_key=API_KEY
       )
       
       # Obtener respuesta
       response_text = response.choices[0].message.content
       
       # Actualizar historial
       chat_history[session_id].append({"role": "user", "content": message})
       chat_history[session_id].append({"role": "assistant", "content": response_text})
       
       # Limitar el tamaño del historial (opcional, para evitar tokens excesivos)
       if len(chat_history[session_id]) > 20:  # Mantener últimos 10 pares pregunta-respuesta
           chat_history[session_id] = chat_history[session_id][-20:]
       
       return response_text
   
   except Exception as e:
       logger.error(f"Error al obtener respuesta: {str(e)}")
       return f"Lo siento, ocurrió un error al procesar tu solicitud: {str(e)}"

def clear_session_history(session_id="default"):
   """Limpia el historial de una sesión específica"""
   if session_id in chat_history:
       chat_history[session_id] = []
       logger.info(f"Historial de la sesión {session_id} eliminado")
   return True

@app.route('/')
def index():
   """Renderiza la página principal"""
   return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
   """Procesa consultas y devuelve respuestas"""
   data = request.json
   user_message = data.get('message', '')
   session_id = data.get('session_id', 'default')  # Usar ID de sesión si está disponible
   
   if not user_message:
       return jsonify({"error": "No se proporcionó un mensaje"}), 400
   
   try:
       response_text = get_response(user_message, session_id)
       
       # Mantener compatibilidad con el frontend anterior
       return jsonify({
           "response": response_text
       })
   
   except Exception as e:
       logger.error(f"Error en el endpoint /query: {str(e)}")
       return jsonify({"error": str(e)}), 500

@app.route('/clear_history', methods=['POST'])
def clear_history():
   """Endpoint para limpiar el historial de chat"""
   data = request.json
   session_id = data.get('session_id', 'default')
   
   success = clear_session_history(session_id)
   
   return jsonify({"success": success})

if __name__ == '__main__':
   app.run(host='127.0.0.1', port=5001, debug=False)