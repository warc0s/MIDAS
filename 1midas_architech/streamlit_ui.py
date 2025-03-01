from __future__ import annotations
from typing import Literal, TypedDict
import asyncio
import os
import streamlit as st
import json
import logfire
from supabase import Client
from openai import AsyncOpenAI

# Configuración de la página: título, layout ancho y sidebar expandido
st.set_page_config(
    page_title="Midas Architech",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="https://github.com/warc0s/MIDAS/blob/main/Extra/Logos/transparentes/architech_trans.png?raw=true"

)

# Agregar estilos CSS personalizados con tonos dorados, animaciones y spinner
st.markdown(
    """
    <style>
        @keyframes float {
            0% { transform: translateY(0px); }
            50% { transform: translateY(-6px); }
            100% { transform: translateY(0px); }
        }
        .message-entrance {
            animation: messageEntrance 0.3s cubic-bezier(0.18, 0.89, 0.32, 1.28) both;
        }
        @keyframes messageEntrance {
            0% { opacity: 0; transform: translateY(20px) scale(0.95); }
            100% { opacity: 1; transform: translateY(0) scale(1); }
        }
        .chatbot-message {
            background: linear-gradient(145deg, #121828 0%, #1A2236 100%);
            box-shadow: 0 8px 32px rgba(18, 24, 40, 0.1);
            border: 1px solid rgba(212, 175, 55, 0.15);
            border-radius: 8px;
            padding: 1rem;
            color: white;
        }
        .user-message {
            background: linear-gradient(45deg, #C4A136 0%, #E5C24C 30%, #F0D675 100%);
            box-shadow: 0 8px 24px rgba(212, 175, 55, 0.2);
            border-radius: 8px;
            padding: 1rem;
            color: black;
            display: inline-block;
            max-width: 100%;
            margin-right: auto;
        }
        .ai-gradient-text {
            background: linear-gradient(135deg, #D4AF37 0%, #FFE55C 100%);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
        }
        .loading-dots:after {
            content: '.';
            animation: dots 1.4s infinite;
        }
        @keyframes dots {
            0%, 20% { content: '.'; }
            40% { content: '..'; }
            60% { content: '...'; }
            80%, 100% { content: ''; }
        }
        .prose :where(code):not(:where([class~="not-prose"] *)) {
            background: rgba(212, 175, 55, 0.15);
            padding: 0.2em 0.4em;
            border-radius: 0.25rem;
        }
        .chatbot-message h3 {
            color: #FFD700 !important;
            font-weight: 600 !important;
            font-size: clamp(1rem, 2vw, 1.125rem) !important;
            margin-bottom: 0.5rem !important;
        }
        .processing-notice {
            display: none;
        }
        .processing-notice.visible {
            display: block;
        }
        /* Spinner animado personalizado */
        .spinner {
            display: inline-block;
            width: 1.5rem;
            height: 1.5rem;
            border: 3px solid rgba(212, 175, 55, 0.3);
            border-radius: 50%;
            border-top-color: #D4AF37;
            animation: spin 1s ease-in-out infinite;
        }
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        /* Responsividad y ajustes generales */
        @media (max-width: 640px) {
            .chat-container {
                width: 100% !important;
                padding: 0.5rem;
            }
            .chatbot-message, .user-message {
                padding: 1rem !important;
            }
            input[type="text"] {
                font-size: 0.9rem;
            }
            header h1 {
                font-size: clamp(1.25rem, 4vw, 1.5rem);
            }
            #clearChat {
                padding: 0.5rem 1rem;
                font-size: 0.9rem;
            }
        }
        @media (min-width: 641px) and (max-width: 768px) {
            .chat-container {
                width: 95% !important;
                margin-left: auto;
                margin-right: auto;
            }
        }
        @media (min-width: 769px) and (max-width: 1024px) {
            .chat-container {
                width: 85% !important;
            }
        }
        @media (min-width: 1025px) {
            .chat-container {
                width: 75% !important;
                max-width: 1200px;
            }
        }
        .text-[15px] {
            font-size: clamp(0.875rem, 1.5vw, 0.9375rem);
        }
        .p-5 {
            padding: clamp(1rem, 3vw, 1.25rem);
        }
        .space-x-4 > * + * {
            margin-left: clamp(0.75rem, 2vw, 1rem);
        }
        .w-9 {
            width: clamp(2rem, 4vw, 2.25rem);
        }
        .h-9 {
            height: clamp(2rem, 4vw, 2.25rem);
        }
        @media (min-width: 768px) {
            .chat-container {
                max-width: 90% !important;
                margin-left: auto;
                margin-right: auto;
            }
        }
        @media (min-width: 1024px) {
            .chat-container {
                max-width: 80% !important;
            }
        }
        /* Custom scrollbar */
        #chatContainer::-webkit-scrollbar {
            width: 8px;
        }
        #chatContainer::-webkit-scrollbar-track {
            background: transparent;
        }
        #chatContainer::-webkit-scrollbar-thumb {
            background: linear-gradient(145deg, #D4AF37, #FFE55C);
            border-radius: 4px;
        }
        #chatContainer::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(145deg, #FFE55C, #D4AF37);
        }
        #chatContainer {
            scrollbar-width: thin;
            scrollbar-color: #D4AF37 transparent;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Importar clases de mensajes
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    UserPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    RetryPromptPart,
    ModelMessagesTypeAdapter
)
from definicion_agentes import pydantic_ai_expert, PydanticAIDeps

# Cargar variables de entorno
from dotenv import load_dotenv
load_dotenv()

openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase: Client = Client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

# Configurar logfire (opcional)
logfire.configure(send_to_logfire='never')

class ChatMessage(TypedDict):
    """Formato de los mensajes enviados a la UI/API."""
    role: Literal['user', 'model']
    timestamp: str
    content: str

def display_message_part(part):
    """
    Muestra una parte del mensaje en la interfaz de Streamlit con animación.
    """
    if part.part_kind == 'system-prompt':
        with st.chat_message("system"):
            st.markdown(f"<div class='message-entrance chatbot-message'><strong>System:</strong> {part.content}</div>", unsafe_allow_html=True)
    elif part.part_kind == 'user-prompt':
        with st.chat_message("user"):
            st.markdown(f"<div class='message-entrance user-message'>{part.content}</div>", unsafe_allow_html=True)
    elif part.part_kind == 'text':
        with st.chat_message("assistant"):
            st.markdown(f"<div class='message-entrance chatbot-message'>{part.content}</div>", unsafe_allow_html=True)

# Configuración de documentación para la aplicación.
# Se actualizan títulos y descripciones para enfatizar la ayuda en sistemas multiagente.
docs_config = {
    "Pydantic AI": {
        "title": "Experto en Pydantic AI docs",
        "description": "Resuelve dudas sobre el framework Pydantic AI basándose en su documentación oficial.",
        "source": "pydantic_ai_docs"
    },
    "CrewAI": {
        "title": "Experto en CrewAI docs",
        "description": "Resuelve dudas sobre el framework CrewAI basándose en su documentación oficial.",
        "source": "crewai_docs"
    },
    "LlamaIndex": {
        "title": "Experto en LlamaIndex docs",
        "description": "Resuelve dudas sobre el framework LlamaIndex basándose en su documentación oficial.",
        "source": "llamaindex_docs"
    },
    "AG2": {
        "title": "Experto en AG2 docs",
        "description": "Resuelve dudas sobre el framework AG2 basándose en su documentación oficial.",
        "source": "ag2_docs"
    }
}

async def safe_stream_text(result):
    """
    Iterador seguro para el streaming que captura y omite
    la excepción cuando se recibe un chunk sin contenido (por ejemplo, tool_calls).
    """
    agen = result.stream_text(delta=True).__aiter__()
    while True:
        try:
            chunk = await agen.__anext__()
        except AssertionError as e:
            if "Expected delta with content" in str(e):
                continue
            else:
                raise
        except StopAsyncIteration:
            break
        yield chunk

async def run_agent_with_streaming(user_input: str):
    """
    Ejecuta el agente con streaming para el prompt user_input,
    manteniendo la conversación en st.session_state.messages.
    Se muestra un spinner animado mientras se procesa la respuesta.
    """
    deps = PydanticAIDeps(
        supabase=supabase,
        openai_client=openai_client,
        docs_source=docs_config[st.session_state.selected_docs]["source"]
    )
    
    # Mostrar spinner animado desde el inicio
    message_placeholder = st.empty()
    spinner_placeholder = st.empty()
    spinner_placeholder.markdown(
        "<div class='processing-notice visible' style='text-align: center; margin-bottom: 1rem;'>"
        "<div class='spinner'></div><br><span class='ai-gradient-text'>Pensando</span>"
        "</div>",
        unsafe_allow_html=True
    )
    
    async with pydantic_ai_expert.run_stream(
        user_input,
        deps=deps,
        message_history=st.session_state.messages[:-1],
    ) as result:
        partial_text = ""
        try:
            async for chunk in safe_stream_text(result):
                if isinstance(chunk, str):
                    partial_text += chunk
                    message_placeholder.markdown(f"<div class='chatbot-message'>{partial_text}</div>", unsafe_allow_html=True)
                elif hasattr(chunk, 'choices') and chunk.choices:
                    choice = chunk.choices[0]
                    if hasattr(choice, 'delta'):
                        delta = choice.delta
                        if hasattr(delta, 'content') and delta.content:
                            partial_text += delta.content
                            message_placeholder.markdown(f"<div class='chatbot-message'>{partial_text}</div>", unsafe_allow_html=True)
                        if hasattr(delta, 'tool_calls') and delta.tool_calls:
                            continue
                elif isinstance(chunk, dict):
                    if 'content' in chunk and chunk['content'] is not None:
                        partial_text += str(chunk['content'])
                        message_placeholder.markdown(f"<div class='chatbot-message'>{partial_text}</div>", unsafe_allow_html=True)
                    elif 'choices' in chunk and chunk['choices']:
                        choice = chunk['choices'][0]
                        if 'delta' in choice and choice['delta'].get('content'):
                            partial_text += choice['delta']['content']
                            message_placeholder.markdown(f"<div class='chatbot-message'>{partial_text}</div>", unsafe_allow_html=True)
                elif chunk is None:
                    break
                else:
                    print(f"Tipo de chunk no manejado: {type(chunk)}")
                    print(f"Contenido del chunk: {chunk}")
                    continue
            spinner_placeholder.empty()
            filtered_messages = [
                msg for msg in result.new_messages() 
                if not (hasattr(msg, 'parts') and any(part.part_kind == 'user-prompt' for part in msg.parts))
            ]
            st.session_state.messages.extend(filtered_messages)
            if partial_text.strip():
                st.session_state.messages.append(
                    ModelResponse(parts=[TextPart(content=partial_text)])
                )
        except Exception as e:
            spinner_placeholder.empty()
            st.error(f"Error durante el streaming: {str(e)}")
            raise

async def main():
    # Sidebar con branding personalizado y selector de documentación
    with st.sidebar:
        st.markdown(
            "<div style='display:flex; justify-content:center'>"
            "<img src='https://github.com/warc0s/MIDAS/blob/main/Extra/Logos/transparentes/architech_trans.png?raw=true' width='150'>"
            "</div>", 
            unsafe_allow_html=True
        )
        st.markdown("<h2 style='text-align: center; color: #FFD700;'>Midas Architech</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #FFFFFF;'>Resuelve dudas de frameworks multiagente usando lenguaje natural</p>", unsafe_allow_html=True)
        st.markdown("---")
        
        # Detectar si se ha cambiado el framework para borrar el chat
        prev_selected = st.session_state.get("selected_docs", None)
        selected_docs = st.selectbox(
            "Selecciona el framework para consulta",
            list(docs_config.keys()),
            index=0
        )
        if prev_selected is not None and prev_selected != selected_docs:
            st.session_state.messages = []  # Auto borrar chat al cambiar de framework
        st.session_state.selected_docs = selected_docs

        # Mostrar el logo correspondiente al framework (placeholder si no se conoce)
        framework_logos = {
            "Pydantic AI": "https://ai.pydantic.dev/img/logo-white.svg",
            "CrewAI": "https://mintlify.s3.us-west-1.amazonaws.com/crewai/crew_only_logo.png",
            "LlamaIndex": "https://cdn.prod.website-files.com/6724087f9af6c1c96461dde4/6724087f9af6c1c96461de13_Group%201326.png",
            "AG2": "https://media.licdn.com/dms/image/v2/D560BAQHYphlMWNdlBg/company-logo_200_200/company-logo_200_200/0/1732554348868/ag2ai_logo?e=2147483647&v=beta&t=x0zpCieHuhWuJ6sKRKit6U3mZm-L42w1aRqHY2606kI"
        }
        logo_url = framework_logos.get(selected_docs, "https://via.placeholder.com/150?text=Logo")
        st.markdown(
            f"<div style='display:flex; justify-content:center'>"
            f"<img src='{logo_url}' width='150'>"
            f"</div>", 
            unsafe_allow_html=True
        )

        st.markdown("<br>", unsafe_allow_html=True)
        
        # Botón para borrar manualmente el chat
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            if st.button("Borrar Chat", type="primary", use_container_width=True):
                st.session_state.messages = []

    config = docs_config[selected_docs]
    
    # Cabecera principal
    st.markdown("<h1 class='ai-gradient-text' style='text-align: center;'>Midas Architech</h1>", unsafe_allow_html=True)
    st.markdown(f"<h2 style='text-align: center;'>{config['title']}</h2>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center;'>{config['description']}</p>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    # Inicializar historial de mensajes si aún no existe
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Mostrar conversación previa con animación
    for msg in st.session_state.messages:
        if isinstance(msg, (ModelRequest, ModelResponse)):
            for part in msg.parts:
                display_message_part(part)

    # Entrada del usuario
    user_input = st.chat_input(f"Escribe aqui tu consulta")

    if user_input:
        st.session_state.messages.append(
            ModelRequest(parts=[UserPromptPart(content=user_input)])
        )
        with st.chat_message("user"):
            st.markdown(f"<div class='user-message message-entrance'>{user_input}</div>", unsafe_allow_html=True)
        with st.chat_message("assistant"):
            await run_agent_with_streaming(user_input)

if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(main())