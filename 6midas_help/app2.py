import os
import sys
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.deepinfra import DeepInfraEmbeddingModel
from llama_index.llms.deepinfra import DeepInfraLLM
import logging

# Configura el logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Configura las credenciales de DeepInfra
load_dotenv()
DEEPINFRA_API_KEY = os.getenv("DEEPINFRA_API_KEY")
if not DEEPINFRA_API_KEY:
    logger.error("DEEPINFRA_API_KEY no está configurada en las variables de entorno.")
    sys.exit("Configuración del servidor incompleta. Contacta al administrador.")

"""
# Configura el modelo de embedding
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-m3"
)
"""
Settings.embed_model = DeepInfraEmbeddingModel(
    model_id="BAAI/bge-m3",  # Usa el ID del modelo personalizado
    api_token=DEEPINFRA_API_KEY,  # Proporciona el token aquí
    normalize=True,  # Normalización opcional
    text_prefix="text: ",  # Prefijo de texto opcional
    query_prefix="query: ",  # Prefijo de consulta opcional
)

Settings.chunk_size = 512

# Configura el modelo LLM
Settings.llm = DeepInfraLLM(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
    api_key=DEEPINFRA_API_KEY,
    temperature=0,
)

# Obtiene el directorio de ejecución actual
execution_dir = os.getcwd()

# Verifica que el archivo .md exista en el directorio de ejecución
md_filename = "info_llmstoryteller.md"
md_path = os.path.join(execution_dir, md_filename)
if not os.path.exists(md_path):
    logger.error(f"El archivo {md_filename} no existe en el directorio de ejecución: {execution_dir}")
    sys.exit("El archivo de información no se encontró en el directorio de ejecución. Contacta al administrador.")

# Lee el archivo info_llmstoryteller.md desde el directorio de ejecución
documents = SimpleDirectoryReader(execution_dir).load_data()

# Filtra para asegurarte de que solo se carga el archivo específico (opcional)
documents = [doc for doc in documents if doc.metadata.get('file_name') == md_filename]

if not documents:
    logger.error(f"No se encontraron documentos con el nombre {md_filename}.")
    sys.exit("No se encontraron documentos válidos para procesar. Contacta al administrador.")

# Crea el índice
logger.info("Creando el índice de vectores. Esto puede tardar unos momentos...")
index = VectorStoreIndex.from_documents(
    documents,
    embed_model=Settings.embed_model  # Asegúrate de usar Settings.embed_model
)
logger.info("Índice creado exitosamente.")

# Configura el motor de consulta
query_engine = index.as_query_engine(llm=Settings.llm)

# Bucle para múltiples consultas
def main():
    print("Bienvenido al sistema de consultas. Escribe 'salir' para terminar.")
    while True:
        try:
            query = input('Por favor, ingrese su consulta: ').strip()
            if query.lower() in ['salir', 'exit', 'quit']:
                print("Saliendo del sistema. ¡Hasta luego!")
                break
            if not query:
                print("La consulta no puede estar vacía. Inténtalo de nuevo.")
                continue
            logger.info(f"Procesando la consulta: {query}")
            response = query_engine.query(query)
            print(f"Respuesta: {response}\n")
        except KeyboardInterrupt:
            print("\nInterrupción detectada. Saliendo del sistema.")
            break
        except Exception as e:
            logger.error(f"Ocurrió un error: {e}")
            print("Ocurrió un error al procesar tu consulta. Por favor, intenta de nuevo.\n")

if __name__ == "__main__":
    main()
