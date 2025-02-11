import os
import sys
import threading
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings, ChatPromptTemplate
from llama_index.embeddings.deepinfra import DeepInfraEmbeddingModel
from llama_index.llms.deepinfra import DeepInfraLLM
import logging

# Configuración del logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicialización de la app Flask
app = Flask(__name__)

# Cargar variables de entorno
load_dotenv()
DEEPINFRA_API_KEY = os.getenv("DEEPINFRA_API_KEY")
if not DEEPINFRA_API_KEY:
    logger.error("DEEPINFRA_API_KEY not configured in environment variables.")
    sys.exit("Server configuration incomplete. Contact administrator.")

# Configuración del modelo de embeddings
Settings.embed_model = DeepInfraEmbeddingModel(
    model_id="BAAI/bge-m3",
    api_token=DEEPINFRA_API_KEY,
    normalize=True,
    text_prefix="text: ",
    query_prefix="query: ",
)

Settings.chunk_size = 512

# Configuración del LLM
Settings.llm = DeepInfraLLM(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
    api_key=DEEPINFRA_API_KEY,
    temperature=0,
)

# Cargar documentos y crear el índice vectorial
execution_dir = os.getcwd()
md_filename = "info_llmstoryteller.md"
md_path = os.path.join(execution_dir, md_filename)

if not os.path.exists(md_path):
    logger.error(f"{md_filename} not found in execution directory: {execution_dir}")
    sys.exit("Information file not found. Contact administrator.")

documents = SimpleDirectoryReader(execution_dir).load_data()
documents = [doc for doc in documents if doc.metadata.get('file_name') == md_filename]

if not documents:
    logger.error(f"No documents found with name {md_filename}.")
    sys.exit("No valid documents found. Contact administrator.")

logger.info("Creating vector index...")
index = VectorStoreIndex.from_documents(documents, embed_model=Settings.embed_model)
logger.info("Index created successfully.")

# --- Configuración de prompts personalizados ---

# Prompt para preguntas (QA)
qa_prompt_str = (
    "Estás asistiendo con dudas sobre el TFM 'Midas'. "
    "Solo responde preguntas relacionadas con este tema. "
    "Si la consulta no está relacionada, responde: 'Lo siento, solo puedo contestar dudas relacionadas con el TFM Midas'.\n"
    "Información de contexto:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Con la información de contexto, responde a la siguiente pregunta: {query_str}\n"
)

# Prompt para refinar la respuesta
refine_prompt_str = (
    "Tenemos la oportunidad de refinar la respuesta original (solo si es necesario) con más contexto a continuación.\n"
    "------------\n"
    "{context_msg}\n"
    "------------\n"
    "Con el nuevo contexto, refina la respuesta original para contestar mejor la pregunta: {query_str}.\n"
    "Si la pregunta no está relacionada con el TFM Midas, mantén la respuesta original o indica que no puedes contestar.\n"
    "Respuesta original: {existing_answer}\n"
)

# Crear las plantillas de prompts utilizando ChatPromptTemplate
chat_text_qa_msgs = [
    (
        "system",
        "Responde solo a preguntas relacionadas con el TFM 'Midas'. "
        "Si la consulta no se relaciona, responde: 'Lo siento, solo puedo contestar dudas relacionadas con el TFM Midas'."
    ),
    ("user", qa_prompt_str),
]
text_qa_template = ChatPromptTemplate.from_messages(chat_text_qa_msgs)

chat_refine_msgs = [
    (
        "system",
        "Responde solo a preguntas relacionadas con el TFM 'Midas'. "
        "Si la consulta no se relaciona, responde: 'Lo siento, solo puedo contestar dudas relacionadas con el TFM Midas'."
    ),
    ("user", refine_prompt_str),
]
refine_template = ChatPromptTemplate.from_messages(chat_refine_msgs)

# Crear el query engine pasando los prompts personalizados
query_engine = index.as_query_engine(
    llm=Settings.llm,
    text_qa_template=text_qa_template,
    refine_template=refine_template
)

# --- Lógica para impedir consultas simultáneas ---
processing_query = False
processing_lock = threading.Lock()
# --------------------------------------------------------

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def handle_query():
    global processing_query
    try:
        data = request.json
        user_input = data.get('message', '').strip()
        
        if not user_input:
            return jsonify({'error': 'La consulta no puede estar vacía'}), 400

        # Verifica si ya se está procesando otra consulta
        with processing_lock:
            if processing_query:
                return jsonify({
                    'error': 'El chatbot ya está procesando una consulta. Por favor, espere.'
                }), 429
            processing_query = True

        logger.info(f"Processing query: {user_input}")
        response = query_engine.query(user_input)
        
        return jsonify({
            'response': str(response),
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        # Aseguramos liberar el bloqueo, incluso si ocurre un error
        with processing_lock:
            processing_query = False

if __name__ == '__main__':
    app.run(debug=True)
