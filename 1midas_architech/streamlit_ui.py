from __future__ import annotations
from typing import Literal, TypedDict
import asyncio
import os
import streamlit as st
import json
import logfire
from supabase import Client
from openai import AsyncOpenAI
import nest_asyncio

# Aplicar nest_asyncio para manejar bucles anidados
nest_asyncio.apply()

# Configuración de la página: título, layout ancho y sidebar expandido
st.set_page_config(
    page_title="Midas Architech | Midas System",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="https://github.com/warc0s/MIDAS/blob/main/Extra/Logos/transparentes/architech_trans.png?raw=true"
)

# Agregar estilos CSS personalizados
st.markdown(
    """
    <style>
        /* Animaciones y efectos visuales */
        @keyframes messageEntrance {
            0% { opacity: 0; transform: translateY(20px); }
            100% { opacity: 1; transform: translateY(0); }
        }
        
        /* Estilo para mensajes del chatbot y usuario */
        .stChatMessage {
            animation: messageEntrance 0.3s ease-out;
        }
        
        /* Estilos para el spinner de carga */
        .stSpinner > div {
            border-top-color: #D4AF37 !important;
        }
        
        /* Ajustes para mejorar la visualización de Markdown */
        .stMarkdown a {
            color: #FFD700;
            text-decoration: underline;
        }
        .stMarkdown pre {
            background-color: #1A1E2E;
            border: 1px solid rgba(212, 175, 55, 0.3);
            border-radius: 5px;
        }
        .stMarkdown code {
            background-color: rgba(212, 175, 55, 0.1);
            color: #FFD700;
            padding: 2px 5px;
            border-radius: 3px;
        }
        
        /* Título principal con gradiente */
        .title-gradient {
            background: linear-gradient(135deg, #D4AF37 0%, #FFE55C 100%);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
            font-weight: bold;
        }
        
        /* Personalización del scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
        }
        ::-webkit-scrollbar-track {
            background: transparent;
        }
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(145deg, #D4AF37, #FFE55C);
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(145deg, #FFE55C, #D4AF37);
        }
        
        /* Mejoras de responsividad */
        @media (max-width: 640px) {
            .stButton button {
                width: 100%;
            }
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

# Inicializar clientes
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase: Client = Client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

# Configurar logfire (opcional)
logfire.configure(send_to_logfire='never')

# Configuración de documentación para la aplicación
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

# Inicializar variables de estado
def init_session_state():
    """Inicializa las variables de estado de la sesión."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "selected_docs" not in st.session_state:
        st.session_state.selected_docs = list(docs_config.keys())[0]
    if "is_processing" not in st.session_state:
        st.session_state.is_processing = False

def get_event_loop():
    """Obtiene un bucle de eventos asyncio funcional."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop
    except RuntimeError:
        # Si no hay bucle en el contexto actual
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop

async def run_agent(user_input: str):
    """Ejecuta el agente con el input del usuario y retorna la respuesta."""
    try:
        deps = PydanticAIDeps(
            supabase=supabase,
            openai_client=openai_client,
            docs_source=docs_config[st.session_state.selected_docs]["source"]
        )
        
        # Ejecutar el agente y obtener la respuesta completa
        result = await pydantic_ai_expert.run(
            user_input,
            deps=deps,
            message_history=st.session_state.messages[:-1],
        )
        
        # Extraer el texto de la respuesta
        response_text = ""
        for msg in result.new_messages():
            if hasattr(msg, 'parts'):
                for part in msg.parts:
                    if part.part_kind == 'text':
                        response_text += part.content
        
        # Actualizar el historial de mensajes con la respuesta completa
        if response_text:
            st.session_state.messages.append(
                ModelResponse(parts=[TextPart(content=response_text)])
            )
        
        return response_text
            
    except Exception as e:
        error_msg = f"Error al procesar tu consulta: {str(e)}"
        print(error_msg)  # Log para depuración
        return error_msg

def display_message_history():
    """Muestra el historial de mensajes en el chat."""
    for msg in st.session_state.messages:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if part.part_kind == 'user-prompt':
                    with st.chat_message("user", avatar="👤"):
                        st.markdown(part.content)
        elif isinstance(msg, ModelResponse):
            for part in msg.parts:
                if part.part_kind == 'text':
                    with st.chat_message("assistant", avatar="🤖"):
                        st.markdown(part.content)

def process_user_input(user_input: str):
    """Procesa la entrada del usuario y genera la respuesta."""
    if st.session_state.is_processing:
        return
    
    st.session_state.is_processing = True
    
    try:
        # Guardar el mensaje del usuario
        user_message = ModelRequest(parts=[UserPromptPart(content=user_input)])
        st.session_state.messages.append(user_message)
        
        # Mostrar el mensaje del usuario
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)
        
        # Mostrar spinner mientras procesamos
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Procesando tu consulta..."):
                loop = get_event_loop()
                response = loop.run_until_complete(run_agent(user_input))
                st.markdown(response)
    
    except Exception as e:
        st.error(f"Error inesperado: {str(e)}")
    
    finally:
        st.session_state.is_processing = False

def app_sidebar():
    """Construye la barra lateral de la aplicación."""
    with st.sidebar:
        # Logo y título
        st.markdown(
            "<div style='display:flex; justify-content:center'>"
            "<img src='https://github.com/warc0s/MIDAS/blob/main/Extra/Logos/transparentes/architech_trans.png?raw=true' width='150'>"
            "</div>", 
            unsafe_allow_html=True
        )
        st.markdown("<h2 style='text-align: center; color: #FFD700;'>Midas Architech</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #FFFFFF;'>Resuelve dudas de frameworks multiagente usando lenguaje natural</p>", unsafe_allow_html=True)
        st.markdown("---")
        
        # Selector de documentación
        prev_selected = st.session_state.selected_docs
        selected_docs = st.selectbox(
            "Selecciona el framework para consulta",
            list(docs_config.keys()),
            index=list(docs_config.keys()).index(st.session_state.selected_docs)
        )
        
        # Resetear chat si se cambia el framework
        if prev_selected != selected_docs:
            st.session_state.selected_docs = selected_docs
            st.session_state.messages = []
            st.rerun()

        # Logo del framework seleccionado
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
        
        # Botón para borrar el chat
        if st.button("Borrar Chat", type="primary", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

def main():
    """Función principal que maneja la interfaz de usuario."""
    # Inicializar estado
    init_session_state()
    
    # Construir la barra lateral
    app_sidebar()

    # Configuración actual
    config = docs_config[st.session_state.selected_docs]
    
    # Cabecera principal
    st.markdown("<h1 class='title-gradient' style='text-align: center;'>Midas Architech</h1>", unsafe_allow_html=True)
    st.markdown(f"<h2 style='text-align: center;'>{config['title']}</h2>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center;'>{config['description']}</p>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    # Mostrar el historial de mensajes
    display_message_history()

    # Campo de entrada del usuario
    user_input = st.chat_input("Escribe aquí tu consulta")
    if user_input and not st.session_state.is_processing:
        process_user_input(user_input)

if __name__ == "__main__":
    main()