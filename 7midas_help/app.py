import os
import sys
import threading
import logging
import re
import unicodedata
import torch
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
from transformers import BertForSequenceClassification, BertTokenizer
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings, ChatPromptTemplate
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

Settings.embed_model = DeepInfraEmbeddingModel(
    model_id="BAAI/bge-m3",
    api_token=DEEPINFRA_API_KEY,
    normalize=True,
    text_prefix="text: ",
    query_prefix="query: ",
)

Settings.chunk_size = 512

Settings.llm = DeepInfraLLM(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
    api_key=DEEPINFRA_API_KEY,
    temperature=0,
)

execution_dir = os.getcwd()
md_filename = "info_midas.md"
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

qa_prompt_str = (
    "Estás asistiendo con dudas y proporcionando respuestas útiles y detalladas. "
    "Aunque tu área principal de especialización es el TFM 'Midas', puedes abordar consultas que sean algo ambiguas o que tengan un vínculo razonable con el tema. "
    "Si la consulta es claramente irrelevante (por ejemplo, si se trata de compras u otros temas no vinculados), responde: 'Lo siento, solo puedo contestar dudas relacionadas con el TFM Midas'.\n"
    "Información de contexto:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Utilizando el contexto proporcionado, responde a la siguiente pregunta: {query_str}\n"
)

chat_text_qa_msgs = [
    (
        "system",
        "Eres un asistente experto capaz de responder consultas variadas, con un enfoque especial en el TFM 'Midas'. Proporciona respuestas útiles y detalladas, y si la consulta es claramente irrelevante (por ejemplo, relacionada con compras u otros temas no vinculados), indica que solo puedes ayudar con temas relacionados con el TFM 'Midas'."
    ),
    ("user", qa_prompt_str),
]
text_qa_template = ChatPromptTemplate.from_messages(chat_text_qa_msgs)

rerank = FlagEmbeddingReranker(model="BAAI/bge-reranker-v2-m3", top_n=3)

query_engine = index.as_query_engine(
    llm=Settings.llm,
    text_qa_template=text_qa_template,
    node_postprocessors=[rerank],
    similarity_top_k=5
)

llm_deepseek = DeepInfraLLM(
    model="deepseek-ai/DeepSeek-V3",
    api_key=DEEPINFRA_API_KEY,
    temperature=0,
)

llm_gemini = Gemini(
    model="models/gemini-2.0-flash",
    api_key=GOOGLE_API_KEY,
    temperature=0,
)

query_engine_deepseek = index.as_query_engine(
    llm=llm_deepseek,
    text_qa_template=text_qa_template,
    node_postprocessors=[rerank],
    similarity_top_k=5
)

query_engine_gemini = index.as_query_engine(
    llm=llm_gemini,
    text_qa_template=text_qa_template,
    node_postprocessors=[rerank],
    similarity_top_k=5
)

processing_query = False
processing_lock = threading.Lock()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def handle_query():
    global processing_query
    try:
        data = request.json
        user_input = data.get('message', '').strip()
        selected_llm = data.get('llm', 'Automatico')
        
        if not user_input:
            return jsonify({'error': 'La consulta no puede estar vacía'}), 400

        # Si se fuerza un LLM, se ignora la clasificación
        if selected_llm != 'Automatico':
            if selected_llm == "Llama 3.3 70B":
                current_engine = query_engine
                llm_usado = 'Llama 3.3 70B'
            elif selected_llm == "DeepSeek V3":
                current_engine = query_engine_deepseek
                llm_usado = 'DeepSeek V3'
            elif selected_llm == "Gemini 2.0 Flash":
                current_engine = query_engine_gemini
                llm_usado = 'Gemini 2.0 Flash'
            else:
                return jsonify({'error': 'Opción de LLM no reconocida.'}), 400
            logger.info(f"LLM forzado: {llm_usado}")
        else:
            # Flujo automático: clasificar el prompt con BERT
            dificultad = clasificar_dificultad(user_input)
            logger.info(f"Prompt classified with difficulty: {dificultad}")
            
            if dificultad == 2:
                return jsonify({
                    'response': "Lo siento, no puedo responder a eso. Si crees que se trata de un error, por favor, reformula la pregunta.",
                    'status': 'success',
                    'llm': 'Bloqueado - PromptAnalysis'
                })
            
            if dificultad == 0:
                current_engine = query_engine
                llm_usado = 'Llama 3.3 70B'
            elif dificultad == 1:
                current_engine = query_engine_deepseek
                llm_usado = 'DeepSeek V3'
            else:
                return jsonify({'error': 'Clasificación de pregunta desconocida.'}), 400

        with processing_lock:
            if processing_query:
                return jsonify({
                    'error': 'El chatbot ya está procesando una consulta. Por favor, espere.'
                }), 429
            processing_query = True

        logger.info(f"Processing query: {user_input}")
        logger.info("Obteniendo embeddings...")
        logger.info("Utilizando el reranker...")
        logger.info("Escribiendo respuesta...")
        response = current_engine.query(user_input)
        
        return jsonify({
            'response': str(response),
            'status': 'success',
            'llm': llm_usado
        })
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        with processing_lock:
            processing_query = False

if __name__ == '__main__':
    app.run(debug=True)