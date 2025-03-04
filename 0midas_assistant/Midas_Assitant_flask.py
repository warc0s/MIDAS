from flask import Flask, render_template, request, jsonify, send_from_directory, session
import os
import uuid
from dotenv import load_dotenv
from litellm import completion
import logging
import time

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv()
API_KEY = os.getenv("API_KEY")
MODEL = os.getenv("MODEL")

# Crear la aplicación Flask
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", os.urandom(24))

# Tiempo de expiración de sesiones (6 horas en segundos)
SESSION_EXPIRY = 6 * 60 * 60

# Estructura de datos para almacenar historiales con caducidad
chat_histories = {}
session_timestamps = {}

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
- Propósito: Sistema RAG (Recuperación Aumentada Generativa) que utiliza Supabase como base de datos vectorial para almacenar y consultar documentación técnica de cuatro frameworks: Pydantic AI, LlamaIndex, CrewAI y AG2.
- Prompt recomendado: "¿Cómo defino un agente en crewai?" o "¿Es adecuado usar AG2 para un sistema rag simple?"
- Capacidades: Proporciona respuestas precisas a consultas técnicas, comprensión contextualizada de documentación, y recuperación inteligente de información relevante. Utiliza Gemini 2.0 Flash como LLM principal.
- Extra: Recuérdale al usuario que debe seleccionar el framework sobre el cual el agente recuperará información de la documentación. No se puede preguntar sobre varios frameworks a la vez. Responde en español a pesar de que la documentación original está en inglés.

### MIDAS DATASET
- Propósito: Genera datasets sintéticos utilizando un sistema multi-agente basado en LLM (Meta Llama 3.3 70B) que coordina la detección de tipos y la generación de datos realistas a través de Faker.
- Prompt recomendado: "Generar 100 registros con columnas: nombre, apellido, edad, ciudad, salario"
- Capacidades: Detección automática del tipo de datos basada en nombres de columnas, configuración de límites para valores numéricos, modificación posterior del dataset generado (añadir/eliminar columnas) y exportación a CSV o Excel.
- Extra: Es necesario especificar explícitamente el número de registros y los nombres de columnas. Los datos se generan con localización española (es_ES) y se pueden establecer valores mínimos y máximos para columnas numéricas para mayor precisión.

### MIDAS PLOT
- Propósito: Sistema basado en CrewAI Flow que genera visualizaciones a partir de un CSV y descripciones en lenguaje natural.
- Prompt recomendado: "Genera una gráfica de barras sobre las calorías de estos cereales" o "Haz un gráfico de barras con las calorías y una línea de puntos con las vitaminas. Pero hazlo solo con los cereales que empiecen por B"
- Capacidades: Genera código matplotlib basado en la descripción del usuario, lo ejecuta en un entorno sandbox (e2b) y devuelve la visualización en formato imagen. 
- Extra: El usuario debe dar un prompt detallado o la gráfica generada será algo simple. También se le puede solicitar que sea de un color específico.

### MIDAS TOUCH
- Propósito: Sistema que automatiza el proceso completo desde la carga de datos hasta el entrenamiento de modelos de machine learning.
- Prompt recomendado: "Entrena un modelo para predecir la columna precio, problema de regresión".
- Capacidades: Análisis automático de datasets, preprocesamiento adaptativo, selección y entrenamiento inteligente de modelos, documentación completa en notebook Jupyter y recuperación ante fallos.
- Extra: En el prompt el usuario debe mencionar explícitamente la columna a predecir, así como definir si es un problema de regresión o clasificación. El sistema utiliza agentes especializados (IntentAgent, DataGuardianAgent, etc.) y está construido sobre Gemini 2.0 Flash.
- Extra2: Especificar tareas/requisitos adicionales en el prompt NO SIRVE DE NADA, NO LO RECOMIENDES.

### MIDAS HELP
- Propósito: Chatbot LLM+RAG+Reranker sobre el repositorio de GitHub de MIDAS para responder dudas técnicas sobre cómo se implementó este TFM.
- Prompts recomendados: "¿Cómo se implementó X componente Midas?" o "¿Qué framework usa Midas Architect?"
- Capacidades: Analiza la pregunta del usuario mediante un clasificador BERT fine-tuned, utiliza embeddings BGE-M3 y un reranker para mejorar los resultados, y selecciona automáticamente entre Llama 3.3 70B (preguntas fáciles) o Gemini 2.0 Flash (preguntas difíciles).
- Extra: Siempre que pregunten sobre detalles específicos de este TFM, di que Midas Help es el que sabe más y les contestará mejor.

### MIDAS TEST
- Propósito: Evaluación exhaustiva de modelos ML en formato joblib mediante agentes conversacionales, analizando rendimiento, robustez y validez de predicciones.
- Prompt recomendado: No necesita prompt específico, solo subir el modelo joblib y pulsar botones.
- Capacidades: Utiliza agentes especializados (Model Analyzer, Performance Tester, Robustness Checker, Output Validator) para evaluar múltiples aspectos del modelo, incluyendo tiempo de carga, uso de memoria y CPU, latencia en diferentes tamaños de batch, resistencia a valores anómalos y consistencia de predicciones.
- Extra: Proporciona una recomendación final ("APTO" o "NO APTO") basada en criterios objetivos y genera un informe detallado en español mediante el modelo Meta-Llama/Llama-3.3-70B-Instruct-Turbo.

### MIDAS ASSISTANT
- Propósito: Ese eres tú. Proporcionas información sobre todos los componentes y recomiendas flujos de trabajo MIDAS.
- Prompt recomendado: "¿Qué componentes debo usar para conseguir X cosa?" o "Dame un prompt efectivo para Midas Plot"
- Capacidades: Orientación general, recomendaciones de prompts y sugerencias de flujos de trabajo completos. Utilizas LiteLLM como framework de abstracción, permitiendo la integración con diferentes modelos de lenguaje como Gemini 2.0 Flash.

### MIDAS DEPLOY
- Propósito: Genera automáticamente interfaces Streamlit personalizadas para modelos de machine learning guardados en formato joblib.
- Prompt recomendado: "Sube tu modelo joblib y describe brevemente su propósito. Por ejemplo: 'Este es un modelo de regresión logística que predice la probabilidad de una condición médica basada en edad, altura y peso del paciente'."
- Capacidades: Utiliza un sistema multi-agente basado en AG2 (Model_Analyzer, UI_Designer, Code_Generator) para analizar el modelo, diseñar una interfaz adaptada y generar código Streamlit ejecutable. Compatible con diversos tipos de modelos ML (clasificadores, regresores, pipelines).
- Extra: Puedes especificar detalles sobre el tipo de usuarios que utilizarán la interfaz (técnicos vs. no técnicos) y mencionar cualquier requisito específico de presentación para las predicciones. Utiliza el modelo Meta-Llama/Llama-3.3-70B-Instruct-Turbo.

## FLUJOS DE TRABAJO TÍPICOS

1. **Flujo completo**: Midas Dataset → Midas Plot → Midas Touch → Midas Test → Midas Deploy
2. **Exploración de datos**: Midas Dataset → Midas Plot
3. **Entrenamiento y evaluación**: Midas Touch → Midas Test
4. **Creación rápida de prototipo**: Midas Dataset → Midas Touch → Midas Deploy

Recuerda siempre proporcionar recomendaciones prácticas y basadas en los componentes de MIDAS, evitando respuestas genéricas.
"""

def generate_session_id():
    """Genera un ID de sesión único"""
    return str(uuid.uuid4())

def cleanup_old_sessions():
    """Limpia sesiones antiguas que han superado el tiempo de expiración"""
    current_time = time.time()
    expired_sessions = []
    
    for session_id, timestamp in session_timestamps.items():
        if current_time - timestamp > SESSION_EXPIRY:
            expired_sessions.append(session_id)
    
    for session_id in expired_sessions:
        if session_id in chat_histories:
            del chat_histories[session_id]
        del session_timestamps[session_id]
        
    if expired_sessions:
        logger.info(f"Limpiadas {len(expired_sessions)} sesiones expiradas")

def get_response(message, session_id):
    """Obtiene una respuesta del LLM usando litellm con memoria de conversación"""
    try:
        logger.info(f"Procesando mensaje para sesión {session_id} usando modelo {MODEL}")
        
        # Actualizar timestamp de la sesión
        session_timestamps[session_id] = time.time()
        
        # Inicializar historial si no existe
        if session_id not in chat_histories:
            chat_histories[session_id] = []
        
        # Construir mensajes con historial
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        # Añadir mensajes anteriores al contexto
        messages.extend(chat_histories[session_id])
        
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
        chat_histories[session_id].append({"role": "user", "content": message})
        chat_histories[session_id].append({"role": "assistant", "content": response_text})
        
        # Limitar el tamaño del historial (para evitar tokens excesivos)
        if len(chat_histories[session_id]) > 20:
            chat_histories[session_id] = chat_histories[session_id][-20:]
        
        # Limpiar sesiones antiguas periódicamente
        if random.random() < 0.1:  # 10% de probabilidad en cada petición
            cleanup_old_sessions()
            
        return response_text
    
    except Exception as e:
        logger.error(f"Error al obtener respuesta: {str(e)}")
        return f"Lo siento, ocurrió un error al procesar tu solicitud: {str(e)}"

def clear_session_history(session_id):
    """Limpia completamente el historial de una sesión específica"""
    if session_id in chat_histories:
        del chat_histories[session_id]
        logger.info(f"Historial de la sesión {session_id} eliminado completamente")
        # Restablecer el timestamp
        session_timestamps[session_id] = time.time()
    return True

@app.route('/')
def index():
    """Renderiza la página principal y asegura que haya un ID de sesión"""
    if 'session_id' not in session:
        session['session_id'] = generate_session_id()
        logger.info(f"Nueva sesión creada: {session['session_id']}")
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    """Procesa consultas y devuelve respuestas"""
    data = request.json
    user_message = data.get('message', '')
    
    # Usar el ID de sesión de Flask o del request
    session_id = session.get('session_id')
    if not session_id:
        session_id = data.get('session_id')
        if not session_id:
            session_id = generate_session_id()
            session['session_id'] = session_id
    
    if not user_message:
        return jsonify({"error": "No se proporcionó un mensaje"}), 400
    
    try:
        response_text = get_response(user_message, session_id)
        
        return jsonify({
            "response": response_text,
            "session_id": session_id  # Devolver ID para que el frontend lo almacene
        })
    
    except Exception as e:
        logger.error(f"Error en el endpoint /query: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/clear_history', methods=['POST'])
def clear_history():
    """Endpoint para limpiar completamente el historial de chat"""
    data = request.json
    
    # Usar el ID de sesión de Flask o del request
    session_id = session.get('session_id')
    if not session_id:
        session_id = data.get('session_id')
        if not session_id:
            return jsonify({"error": "No se encontró ID de sesión"}), 400
    
    success = clear_session_history(session_id)
    
    return jsonify({
        "success": success,
        "message": "Historial eliminado completamente"
    })

if __name__ == '__main__':
    # Para importación: necesitas "import random" al inicio
    import random
    app.run(host='127.0.0.1', port=5001, debug=False)
