import os
import sys
import threading
import logging
import re
import unicodedata
import torch
import uuid
import random
from datetime import datetime, timedelta
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
from transformers import BertForSequenceClassification, BertTokenizer
from llama_index.core import (
    SimpleDirectoryReader, 
    VectorStoreIndex, 
    Settings, 
    ChatPromptTemplate,
    StorageContext,
    load_index_from_storage
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.deepinfra import DeepInfraEmbeddingModel
from llama_index.llms.deepinfra import DeepInfraLLM
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from llama_index.llms.gemini import Gemini

# -------------------------- Integración del BERT para clasificación del prompt --------------------------

def clean_text(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

bert_model = BertForSequenceClassification.from_pretrained("prompt_analysis")
bert_tokenizer = BertTokenizer.from_pretrained("prompt_analysis")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model.to(device)

def clasificar_dificultad(texto):
    texto_limpio = clean_text(texto)
    inputs = bert_tokenizer(texto_limpio, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = bert_model(**inputs)
    logits = outputs.logits
    prediccion = torch.argmax(logits, dim=1).item()
    return prediccion

# ---------------------------------------------------------------------------------------------------

# ------------- Sistema de memoria para conversaciones -------------
conversation_history = {}  # {session_id: [lista de mensajes]}
last_activity = {}  # {session_id: timestamp}
SESSION_TIMEOUT = timedelta(hours=12)  # Sesiones inactivas por más de 12 horas se eliminarán

def clean_inactive_sessions():
    """Elimina sesiones inactivas para liberar memoria"""
    current_time = datetime.now()
    inactive_sessions = []
    
    for session_id, timestamp in list(last_activity.items()):
        if current_time - timestamp > SESSION_TIMEOUT:
            inactive_sessions.append(session_id)
    
    for session_id in inactive_sessions:
        if session_id in conversation_history:
            del conversation_history[session_id]
        if session_id in last_activity:
            del last_activity[session_id]
        if session_id in processing_queries:
            del processing_queries[session_id]
    
    if inactive_sessions:
        #logger.info(f"Eliminadas {len(inactive_sessions)} sesiones inactivas")
        pass

def get_conversation_history(session_id):
    """Obtiene o crea un historial de conversación para el ID de sesión dado"""
    if not session_id:
        session_id = str(uuid.uuid4())
    
    if session_id not in conversation_history:
        conversation_history[session_id] = []
        #logger.info(f"Creada nueva sesión con ID: {session_id}")
    
    # Actualizar timestamp de última actividad
    last_activity[session_id] = datetime.now()
    
    return session_id, conversation_history[session_id]

def add_to_history(session_id, role, content):
    """Añade un mensaje al historial de conversación"""
    if session_id in conversation_history:
        conversation_history[session_id].append({
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat()
        })
        # Actualizar timestamp de última actividad
        last_activity[session_id] = datetime.now()
        #logger.info(f"Añadido mensaje de {role} a sesión {session_id}")
    
def format_conversation_history(history, max_messages=5):
    """Formatea el historial de conversación para incluirlo en el prompt"""
    if not history:
        return ""
        
    # Limitar a los últimos N mensajes para no sobrecargar el contexto
    recent_history = history[-max_messages:] if len(history) > max_messages else history
    formatted_history = ""
    
    for message in recent_history:
        role = "Usuario" if message['role'] == 'user' else "Asistente"
        formatted_history += f"{role}: {message['content']}\n\n"
    
    return formatted_history

def modify_query_with_history(history, query_str):
    """Modifica la consulta para incluir el historial de conversación relevante"""
    if not history:
        return query_str
    
    # Formatear el historial de conversación
    formatted_history = format_conversation_history(history)
    
    # Construir la consulta mejorada que incluye el historial
    enhanced_query = (
        f"HISTORIAL DE CONVERSACIÓN RELEVANTE:\n"
        f"{formatted_history}\n"
        f"CONSULTA ACTUAL: {query_str}\n\n"
        f"Responde a la CONSULTA ACTUAL teniendo en cuenta el HISTORIAL DE CONVERSACIÓN RELEVANTE. "
        f"Si la consulta hace referencia a elementos mencionados anteriormente en la conversación, "
        f"asegúrate de contextualizar tu respuesta correctamente."
    )
    
    return enhanced_query
# -------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

load_dotenv()
DEEPINFRA_API_KEY = os.getenv("DEEPINFRA_API_KEY")
if not DEEPINFRA_API_KEY:
    logger.error("DEEPINFRA_API_KEY not configured in environment variables.")
    sys.exit("Server configuration incomplete. Contact administrator.")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    logger.error("GOOGLE_API_KEY not configured in environment variables.")
    sys.exit("Server configuration incomplete. Contact administrator.")

# Configuración del modelo de embeddings
Settings.embed_model = DeepInfraEmbeddingModel(
    model_id="BAAI/bge-m3",
    api_token=DEEPINFRA_API_KEY,
    normalize=True,
    text_prefix="text: ",
    query_prefix="query: ",
)

# Directorio para almacenar datos de índice
INDEX_STORAGE_PATH = os.path.join(os.getcwd(), "index_storage")

# Mejorado: chunking más inteligente con solapamiento
node_parser = SentenceSplitter(
    chunk_size=512,
    chunk_overlap=100,  # Overlap del 20%
    paragraph_separator="\n\n",
    secondary_chunking_regex="(?<=\. )"  # Divide por oraciones como respaldo
)

# Configuramos LLMs
Settings.llm = DeepInfraLLM(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
    api_key=DEEPINFRA_API_KEY,
    temperature=0.3
)

llm_gemini = Gemini(
    model="models/gemini-2.0-flash",
    api_key=GOOGLE_API_KEY,
    temperature=0.3
)

# Verificación de documentos
execution_dir = os.getcwd()
md_filename = "info_midas.md"
md_path = os.path.join(execution_dir, md_filename)

if not os.path.exists(md_path):
    logger.error(f"{md_filename} not found in execution directory: {execution_dir}")
    sys.exit("Information file not found. Contact administrator.")

# Función para crear o cargar índice
def get_or_create_index():
    # Intentar cargar el índice desde almacenamiento
    if os.path.exists(INDEX_STORAGE_PATH):
        try:
            #logger.info("Loading index from storage...")
            storage_context = StorageContext.from_defaults(persist_dir=INDEX_STORAGE_PATH)
            index = load_index_from_storage(storage_context=storage_context)
            #logger.info("Index loaded successfully.")
            return index
        except Exception as e:
            logger.warning(f"Error loading index: {e}. Creating new index...")
    
    # Si no existe o hay error, crear nuevo índice
    #logger.info("Creating new vector index...")
    documents = SimpleDirectoryReader(execution_dir).load_data()
    documents = [doc for doc in documents if doc.metadata.get('file_name') == md_filename]
    
    if not documents:
        logger.error(f"No documents found with name {md_filename}.")
        sys.exit("No valid documents found. Contact administrator.")
    
    # Usar el node_parser mejorado
    index = VectorStoreIndex.from_documents(
        documents, 
        embed_model=Settings.embed_model,
        transformations=[node_parser]
    )
    
    # Persistir el índice
    index.storage_context.persist(persist_dir=INDEX_STORAGE_PATH)
    #logger.info("Index created and persisted successfully.")
    return index

# Obtener o crear índice
index = get_or_create_index()

# --- Configuración de prompts personalizados mejorados ---

qa_prompt_str = (
    "Eres un asistente de IA especializado en el TFM (trabajo de fun de master) llamado 'Midas'. Tu tarea es proporcionar respuestas precisas, útiles y detalladas.\n\n"
    "INSTRUCCIONES DE SÍNTESIS:\n"
    "1. Analiza cuidadosamente todo el contexto proporcionado.\n"
    "2. Identifica las piezas de información más relevantes para la consulta específica.\n"
    "3. Integra coherentemente la información, evitando repeticiones innecesarias.\n"
    "4. Si la información en el contexto es insuficiente, indícalo claramente.\n"
    "5. Si hay información contradictoria, menciona las diferentes perspectivas.\n"
    "6. Organiza tu respuesta de forma lógica, empezando con los puntos más importantes.\n\n"
    "RESPUESTA:\n"
    "- Responde de manera clara y directa.\n"
    "- Incluye detalles específicos del TFM Midas cuando sea relevante.\n"
    "- Si la consulta es ambigua pero relacionable con el tema, intenta interpretar la intención del usuario.\n"
    "Información de contexto:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Consulta: {query_str}\n\n"
)

chat_text_qa_msgs = [
    (
        "system",
        "Eres un asistente experto especializado en el TFM 'Midas'. Tu objetivo es proporcionar respuestas precisas, "
        "informativas y bien estructuradas basadas en la información disponible. Utiliza solo los datos proporcionados "
        "en el contexto para responder, evitando invenciones o suposiciones no respaldadas. Si la información es insuficiente, "
        "indícalo claramente en lugar de elaborar respuestas imprecisas. Si la consulta es completamente irrelevante con el sistema 'Midas', "
        "indica amablemente que solo puedes responder sobre ese tema específico. Pero ante la duda, opta por contestar la consulta."
    ),
    ("user", qa_prompt_str),
]
text_qa_template = ChatPromptTemplate.from_messages(chat_text_qa_msgs)

# Mejorado: reranker para mejor selección de contexto relevante
rerank = FlagEmbeddingReranker(
    model="BAAI/bge-reranker-v2-m3", 
    top_n=10  
)

# Crear query engines con mejoras
def create_enhanced_query_engine(llm_model, similarity_top_k=30):
    return index.as_query_engine(
        llm=llm_model,
        text_qa_template=text_qa_template,
        node_postprocessors=[rerank],
        similarity_top_k=similarity_top_k
    )

# Crear los query engines mejorados
query_engine = create_enhanced_query_engine(Settings.llm)
query_engine_gemini = create_enhanced_query_engine(llm_gemini)

# Sistema de concurrencia por sesión (reemplaza el sistema global)
processing_queries = {}  # {session_id: is_processing}
processing_lock = threading.Lock()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def handle_query():
    try:
        data = request.json
        user_input = data.get('message', '').strip()
        selected_llm = data.get('llm', 'Automatico')
        session_id = data.get('session_id', '')
        
        if not user_input:
            return jsonify({'error': 'La consulta no puede estar vacía'}), 400

        # Limpieza periódica de sesiones inactivas (5% de probabilidad en cada petición)
        if random.random() < 0.05:
            clean_inactive_sessions()

        # Obtener o crear historial de conversación
        session_id, history = get_conversation_history(session_id)
        
        # Añadir la consulta actual al historial
        add_to_history(session_id, 'user', user_input)
        #logger.info(f"Consulta registrada. Sesión {session_id} tiene {len(history)} mensajes")

        # Verificar si esta sesión específica ya está procesando una consulta
        with processing_lock:
            if session_id in processing_queries and processing_queries[session_id]:
                return jsonify({
                    'error': 'Ya estamos procesando tu consulta anterior. Por favor, espera un momento.'
                }), 429
            processing_queries[session_id] = True

        try:
            # Selección de LLM basada en la entrada del usuario
            if selected_llm != 'Automatico':
                if selected_llm == "Llama 3.3 70B":
                    current_engine = query_engine
                    llm_usado = 'Llama 3.3 70B'
                elif selected_llm == "Gemini 2.0 Flash":
                    current_engine = query_engine_gemini
                    llm_usado = 'Gemini 2.0 Flash'
                else:
                    # Liberar el bloqueo antes de retornar error
                    with processing_lock:
                        processing_queries[session_id] = False
                    return jsonify({'error': 'Opción de LLM no reconocida.'}), 400
                #logger.info(f"LLM forzado: {llm_usado}")
            else:
                # Flujo automático: clasificar el prompt con BERT
                dificultad = clasificar_dificultad(user_input)
                #logger.info(f"Prompt classified with difficulty: {dificultad}")
                
                if dificultad == 2:
                    response_text = "Lo siento, no puedo responder a eso. Si crees que se trata de un error, por favor, reformula la pregunta."
                    # Añadir la respuesta al historial
                    add_to_history(session_id, 'assistant', response_text)
                    
                    # Liberar el bloqueo
                    with processing_lock:
                        processing_queries[session_id] = False
                        
                    response_data = {
                        'response': response_text,
                        'status': 'success',
                        'llm': 'Bloqueado - PromptAnalysis',
                        'session_id': session_id
                    }
                    return jsonify(response_data)
                
                if dificultad == 0:
                    current_engine = query_engine
                    llm_usado = 'Llama 3.3 70B'
                elif dificultad == 1:
                    current_engine = query_engine_gemini
                    llm_usado = 'Gemini 2.0 Flash'
                else:
                    # Liberar el bloqueo antes de retornar error
                    with processing_lock:
                        processing_queries[session_id] = False
                    return jsonify({'error': 'Clasificación de pregunta desconocida.'}), 400

            #logger.info(f"Procesando consulta con contexto. Sesión: {session_id}")
            
            # Obtener historial previo (excluyendo la consulta actual recién añadida)
            previous_history = history[:-1] if len(history) > 1 else []
            
            # Modificar la consulta para incluir el historial relevante
            enhanced_query = modify_query_with_history(previous_history, user_input)
            #logger.info(f"Consulta mejorada creada con contexto. Longitud historia: {len(previous_history)}")
            
            # Realizar la consulta con el contexto mejorado
            response = current_engine.query(enhanced_query)
            response_text = str(response)
            
            # Añadir la respuesta al historial
            add_to_history(session_id, 'assistant', response_text)
            #logger.info(f"Respuesta añadida. Ahora la sesión tiene {len(history)} mensajes")
            
            response_data = {
                'response': response_text,
                'status': 'success',
                'llm': llm_usado,
                'session_id': session_id
            }
            
            return jsonify(response_data)
            
        finally:
            # Asegurar que el bloqueo siempre se libere
            with processing_lock:
                if session_id in processing_queries:
                    processing_queries[session_id] = False
        
    except Exception as e:
        # En caso de excepción, asegurarse de liberar el bloqueo
        if 'session_id' in locals() and session_id:
            with processing_lock:
                if session_id in processing_queries:
                    processing_queries[session_id] = False
        logger.error(f"Error procesando consulta: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/clear_history', methods=['POST'])
def clear_conversation_history():
    try:
        data = request.json
        session_id = data.get('session_id', '')
        
        if session_id and session_id in conversation_history:
            conversation_history[session_id] = []
            #logger.info(f"Historial borrado para sesión {session_id}")
            # Actualizar timestamp de última actividad
            last_activity[session_id] = datetime.now()
            return jsonify({
                'status': 'success', 
                'message': 'Historial de conversación borrado',
                'session_id': session_id
            })
        return jsonify({
            'status': 'success', 
            'message': 'No hay historial para borrar',
            'session_id': session_id if session_id else str(uuid.uuid4())
        })
    except Exception as e:
        logger.error(f"Error borrando historial: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5008, debug=False)