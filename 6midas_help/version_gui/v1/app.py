import os
import sys
import threading
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.embeddings.deepinfra import DeepInfraEmbeddingModel
from llama_index.llms.deepinfra import DeepInfraLLM
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Load environment variables
load_dotenv()
DEEPINFRA_API_KEY = os.getenv("DEEPINFRA_API_KEY")
if not DEEPINFRA_API_KEY:
    logger.error("DEEPINFRA_API_KEY not configured in environment variables.")
    sys.exit("Server configuration incomplete. Contact administrator.")

# Configure embedding model
Settings.embed_model = DeepInfraEmbeddingModel(
    model_id="BAAI/bge-m3",
    api_token=DEEPINFRA_API_KEY,
    normalize=True,
    text_prefix="text: ",
    query_prefix="query: ",
)

Settings.chunk_size = 512

# Configure LLM
Settings.llm = DeepInfraLLM(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
    api_key=DEEPINFRA_API_KEY,
    temperature=0,
)

# Load documents and create index
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

query_engine = index.as_query_engine(llm=Settings.llm)

# --- Nueva lógica para impedir consultas simultáneas ---
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
                # Se informa que ya hay una consulta en curso
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
