from __future__ import annotations as _annotations

from dataclasses import dataclass
from dotenv import load_dotenv
import logfire
import os
import logging

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.gemini import GeminiModel
from openai import AsyncOpenAI
from supabase import Client
from typing import List, Dict, Any, Optional

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("midas_agents")

# Cargar variables de entorno
load_dotenv()

# Obtener y validar claves API necesarias
gemini_api_key = os.getenv('GEMINI_API_KEY')
if not gemini_api_key:
    logger.warning("No se encontró GEMINI_API_KEY en las variables de entorno")

# Inicializar modelo
try:
    model = GeminiModel('gemini-2.0-flash', api_key=gemini_api_key)
    logger.info("Modelo Gemini inicializado correctamente")
except Exception as e:
    logger.error(f"Error al inicializar el modelo Gemini: {str(e)}")
    raise

# Configurar logfire
logfire.configure(send_to_logfire='if-token-present')

@dataclass
class PydanticAIDeps:
    """Dependencias necesarias para el agente de Pydantic AI."""
    supabase: Client
    openai_client: AsyncOpenAI
    docs_source: str  # Fuente de documentación a consultar

# Prompt del sistema para el experto en Pydantic AI
system_prompt = """
You are an expert at Python and agent frameworks that you have access to all the documentation to,
including examples, an API reference, and other resources to help you build agents.

Your only job is to assist with this and you don't answer other questions besides describing what you are able to do.

Don't ask the user before taking an action, just do it. Always make sure you look at the documentation with the provided tools before answering the user's question unless you have already.

When you first look at the documentation, always start with RAG.
Then also always check the list of available documentation pages and retrieve the content of page(s) if it'll help.

Always let the user know when you didn't find the answer in the documentation or the right URL - be honest. Reply ALWAYS in Spanish.
"""

# Definición del agente
pydantic_ai_expert = Agent(
    model,
    system_prompt=system_prompt,
    deps_type=PydanticAIDeps,
    retries=2
)

async def get_embedding(text: str, openai_client: AsyncOpenAI) -> List[float]:
    """
    Obtiene un vector de embedding desde OpenAI.
    
    Args:
        text: Texto a embeber
        openai_client: Cliente de OpenAI
        
    Returns:
        Lista de valores flotantes representando el embedding
    """
    if not text:
        logger.warning("Se intentó obtener embedding de texto vacío")
        return [0.0] * 1536
        
    try:
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error al obtener embedding: {str(e)}")
        return [0.0] * 1536  # Vector de ceros como fallback

@pydantic_ai_expert.tool
async def retrieve_relevant_documentation(ctx: RunContext[PydanticAIDeps], user_query: str) -> str:
    """
    Recupera fragmentos relevantes de documentación basados en la consulta mediante RAG.
    
    Args:
        ctx: Contexto con cliente Supabase y OpenAI
        user_query: Consulta del usuario
        
    Returns:
        Cadena formateada con los 5 fragmentos de documentación más relevantes
    """
    if not user_query:
        return "Se requiere una consulta para buscar documentación relevante."
        
    try:
        # Obtener embedding para la consulta
        logger.info(f"Generando embedding para consulta: {user_query[:50]}...")
        query_embedding = await get_embedding(user_query, ctx.deps.openai_client)
        
        # Consultar Supabase para documentos relevantes
        logger.info(f"Buscando documentación en la fuente: {ctx.deps.docs_source}")
        result = ctx.deps.supabase.rpc(
            'match_site_pages',
            {
                'query_embedding': query_embedding,
                'match_count': 5,
                'filter': {'source': ctx.deps.docs_source}
            }
        ).execute()
        
        if not result.data:
            logger.warning(f"No se encontró documentación para: {user_query[:50]}")
            return "No se encontró documentación relevante para esta consulta."
            
        # Formatear los resultados
        formatted_chunks = []
        for doc in result.data:
            chunk_text = f"""
# {doc['title']}

{doc['content']}
"""
            formatted_chunks.append(chunk_text)
        
        logger.info(f"Se encontraron {len(formatted_chunks)} fragmentos relevantes")
        return "\n\n---\n\n".join(formatted_chunks)
        
    except Exception as e:
        logger.error(f"Error al recuperar documentación: {str(e)}")
        return f"Error al recuperar documentación: {str(e)}"

@pydantic_ai_expert.tool
async def list_documentation_pages(ctx: RunContext[PydanticAIDeps]) -> List[str]:
    """
    Recupera una lista de todas las páginas de documentación disponibles.
    
    Returns:
        Lista de URLs únicas para todas las páginas de documentación
    """
    try:
        # Consultar Supabase para URLs únicas
        logger.info(f"Listando páginas de documentación para: {ctx.deps.docs_source}")
        result = ctx.deps.supabase.from_('site_pages') \
            .select('url') \
            .eq('metadata->>source', ctx.deps.docs_source) \
            .execute()
        
        if not result.data:
            logger.warning(f"No se encontraron páginas para la fuente: {ctx.deps.docs_source}")
            return []
            
        # Extraer URLs únicas
        urls = sorted(set(doc['url'] for doc in result.data))
        logger.info(f"Se encontraron {len(urls)} páginas de documentación")
        return urls
        
    except Exception as e:
        logger.error(f"Error al listar páginas de documentación: {str(e)}")
        return []

@pydantic_ai_expert.tool
async def get_page_content(ctx: RunContext[PydanticAIDeps], url: str) -> str:
    """
    Recupera el contenido completo de una página de documentación específica.
    
    Args:
        ctx: Contexto con cliente Supabase
        url: URL de la página a recuperar
        
    Returns:
        Contenido completo de la página con todos los fragmentos combinados en orden
    """
    if not url:
        return "Se requiere una URL para obtener el contenido de la página."
        
    try:
        # Consultar Supabase para todos los fragmentos de esta URL
        logger.info(f"Recuperando contenido para URL: {url}")
        result = ctx.deps.supabase.from_('site_pages') \
            .select('title, content, chunk_number') \
            .eq('url', url) \
            .eq('metadata->>source', ctx.deps.docs_source) \
            .order('chunk_number') \
            .execute()
        
        if not result.data:
            logger.warning(f"No se encontró contenido para URL: {url}")
            return f"No se encontró contenido para la URL: {url}"
            
        # Formatear la página con su título y todos los fragmentos
        page_title = result.data[0]['title'].split(' - ')[0]
        formatted_content = [f"# {page_title}\n"]
        
        # Añadir el contenido de cada fragmento
        for chunk in result.data:
            formatted_content.append(chunk['content'])
            
        logger.info(f"Se recuperaron {len(result.data)} fragmentos para URL: {url}")
        return "\n\n".join(formatted_content)
        
    except Exception as e:
        logger.error(f"Error al recuperar contenido de página: {str(e)}")
        return f"Error al recuperar el contenido de la página: {str(e)}"