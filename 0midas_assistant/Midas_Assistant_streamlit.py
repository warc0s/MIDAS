#!/usr/bin/env python3
"""
Midas Assistant Streamlit - Una interfaz web para el chatbot basado en LiteLLM que proporciona información 
sobre los componentes de MIDAS (Multi-agent Intelligent Data Automation System).
"""

import os
import sys
import importlib.util
import streamlit as st
from typing import List, Dict
from dotenv import load_dotenv
from litellm import completion
import re

# Cargar variables de entorno
load_dotenv()
API_KEY = os.getenv("API_KEY")
MODEL = os.getenv("MODEL", "gemini/gemini-2.0-flash")

# Importar el SYSTEM_PROMPT desde el archivo original
def import_system_prompt_from_cli():
    try:
        # Importar el archivo Midas_Assistant_cli.py para obtener el SYSTEM_PROMPT
        spec = importlib.util.spec_from_file_location("midas_cli", "Midas_Assistant_cli.py")
        midas_cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(midas_cli)
        
        # Obtener el SYSTEM_PROMPT del módulo importado
        return midas_cli.SYSTEM_PROMPT
    except Exception as e:
        st.error(f"Error al importar SYSTEM_PROMPT: {str(e)}")
        # Sistema de respaldo en caso de error
        return """
        Eres Midas Assistant, un asistente especializado en el sistema MIDAS (Multi-agent Intelligent Data Automation System).
        Tu objetivo es ayudar a los usuarios a comprender y utilizar eficientemente los componentes del sistema MIDAS.
        """

# Obtener el SYSTEM_PROMPT del archivo CLI
SYSTEM_PROMPT = import_system_prompt_from_cli()

def format_response(text: str) -> str:
    """Formatea un texto para resaltar componentes MIDAS usando markdown de Streamlit"""
    # Resaltar los nombres de los componentes
    components = ["MIDAS ARCHITECT", "MIDAS DATASET", "MIDAS PLOT", "MIDAS TOUCH", 
                  "MIDAS HELP", "MIDAS TEST", "MIDAS ASSISTANT", "MIDAS DEPLOY"]
    
    for component in components:
        text = re.sub(f"({component})", r"**\1**", text, flags=re.IGNORECASE)
    
    # Resaltar citas o ejemplos de prompts
    text = re.sub(r'(".*?")', r"*\1*", text)
    
    # Resaltar pasos numerados
    text = re.sub(r'(\d+\.\s)', r"**\1**", text)
    
    return text

def get_streaming_response(prompt: str, chat_history: List[Dict] = None):
    """
    Obtiene una respuesta del modelo con streaming para Streamlit
    """
    if chat_history is None:
        chat_history = []
    
    try:
        # Mostrar spinner con estilo personalizado
        with st.spinner(""):
            # Usamos un contenedor vacío para mostrar nuestro propio spinner estilizado
            thinking_container = st.empty()
            thinking_container.markdown(
                """
                <div class="thinking-container">
                    <span class="thinking-text">Pensando<span class="thinking-dots"></span></span>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
            # Preparar la llamada al modelo
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                *chat_history,
                {"role": "user", "content": prompt}
            ]
            
            # Iniciar solicitud
            response_stream = completion(
                model=MODEL,
                messages=messages,
                api_key=API_KEY,
                stream=True
            )
            
            # Limpiar el contenedor de "pensando" una vez que estamos listos para responder
            thinking_container.empty()
        
        # Una vez que tenemos el stream, creamos el mensaje del asistente
        with st.chat_message("assistant", avatar="🤖"):
            message_placeholder = st.empty()
            full_response = ""
            
            # Procesar cada fragmento de la respuesta
            for chunk in response_stream:
                content = chunk.choices[0].delta.content or ""
                if content:
                    full_response += content
                    # Actualizar el placeholder con la respuesta formateada
                    message_placeholder.markdown(format_response(full_response) + "▌")
            
            # Actualización final sin el cursor
            message_placeholder.markdown(format_response(full_response))
        
        return full_response
        
    except Exception as e:
        error_msg = f"Error al comunicarse con el modelo: {str(e)}"
        st.error(error_msg)
        return error_msg

def initialize_session_state():
    """Inicializa variables en el estado de la sesión"""
    if "messages" not in st.session_state:
        st.session_state.messages = []

# Estilos CSS personalizados basados en Midas Help
def load_custom_css():
    st.markdown("""
    <style>
        /* Estilos globales y fuentes */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Animaciones */
        @keyframes float {
            0% { transform: translateY(0px); }
            50% { transform: translateY(-6px); }
            100% { transform: translateY(0px); }
        }
        
        @keyframes messageEntrance {
            0% { opacity: 0; transform: translateY(20px) scale(0.95); }
            100% { opacity: 1; transform: translateY(0) scale(1); }
        }
        
        @keyframes ellipsis {
            0% { content: '.'; }
            33% { content: '..'; }
            66% { content: '...'; }
            100% { content: ''; }
        }
        
        /* Estilos base para Streamlit */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1200px;
        }
        
        /* Título y encabezado */
        h1 {
            color: #FFD700 !important;
            font-weight: 600 !important;
            margin-bottom: 1rem !important;
            font-family: 'Inter', sans-serif !important;
        }
        
        /* Estilo general de párrafos */
        p {
            font-family: 'Inter', sans-serif !important;
            font-size: 1rem !important;
            line-height: 1.6 !important;
        }
        
        /* Personalización de los mensajes */
        .stChatMessage {
            animation: messageEntrance 0.3s cubic-bezier(0.18, 0.89, 0.32, 1.28) both;
            margin-bottom: 1.5rem !important;
            border-radius: 1rem !important;
        }
        
        /* Mensajes del bot */
        .stChatMessage[data-testid="StChatMessage"] .stChatMessageContent:has(div[data-testid="chatAvatarIcon-assistant"]) {
            background: linear-gradient(145deg, #121828 0%, #1A2236 100%) !important;
            box-shadow: 0 8px 32px rgba(18, 24, 40, 0.1) !important;
            border: 1px solid rgba(212, 175, 55, 0.15) !important;
            padding: 1.25rem !important;
            border-radius: 1rem !important;
        }
        
        /* Mensajes del usuario */
        .stChatMessage[data-testid="StChatMessage"] .stChatMessageContent:has(div[data-testid="chatAvatarIcon-user"]) {
            background: linear-gradient(45deg, #D4AF37 0%, #FFD700 30%, #FFE55C 100%) !important;
            box-shadow: 0 8px 24px rgba(212, 175, 55, 0.2) !important;
            padding: 1.25rem !important;
            border-radius: 1rem !important;
        }
        
        /* Color de texto en mensajes del usuario */
        .stChatMessage[data-testid="StChatMessage"] .stChatMessageContent:has(div[data-testid="chatAvatarIcon-user"]) p {
            color: #121828 !important;
            font-weight: 500 !important;
        }
        
        /* Color de texto en mensajes del asistente */
        .stChatMessage[data-testid="StChatMessage"] .stChatMessageContent:has(div[data-testid="chatAvatarIcon-assistant"]) p {
            color: #f0f0f0 !important;
        }
        
        /* Estilos para negritas en mensajes del asistente */
        .stChatMessage[data-testid="StChatMessage"] .stChatMessageContent:has(div[data-testid="chatAvatarIcon-assistant"]) strong {
            color: #FFD700 !important;
            font-weight: 600 !important;
        }
        
        /* Estilos para cursivas en mensajes del asistente */
        .stChatMessage[data-testid="StChatMessage"] .stChatMessageContent:has(div[data-testid="chatAvatarIcon-assistant"]) em {
            color: #FFE55C !important;
        }
        
        /* Input de chat */
        .stChatInputContainer {
            background: rgba(212, 175, 55, 0.05) !important;
            border: 1px solid rgba(212, 175, 55, 0.2) !important;
            border-radius: 1rem !important;
            padding: 0.5rem !important;
        }
        
        .stChatInputContainer input[type="text"] {
            color: #f0f0f0 !important;
            font-size: 1rem !important;
            font-family: 'Inter', sans-serif !important;
        }
        
        .stChatInputContainer button[data-testid="sendButton"] {
            background: linear-gradient(45deg, #D4AF37 0%, #FFD700 100%) !important;
            color: #121828 !important;
        }
        
        /* Avatar del asistente */
        div[data-testid="chatAvatarIcon-assistant"] {
            background: linear-gradient(135deg, #D4AF37 0%, #FFE55C 100%) !important;
            color: #121828 !important;
            font-weight: bold !important;
            border: none !important;
        }
        
        /* Avatar del usuario */
        div[data-testid="chatAvatarIcon-user"] {
            background: #121828 !important;
            color: #FFD700 !important;
            border: 2px solid #D4AF37 !important;
        }
        
        /* Estilo para el indicador "Pensando" */
        .thinking-container {
            display: flex;
            align-items: center;
            justify-content: center;
            background: linear-gradient(145deg, #121828 0%, #1A2236 100%);
            border: 1px solid rgba(212, 175, 55, 0.15);
            border-radius: 1rem;
            padding: 1rem;
            margin: 1rem 0;
            box-shadow: 0 8px 32px rgba(18, 24, 40, 0.1);
            animation: messageEntrance 0.3s cubic-bezier(0.18, 0.89, 0.32, 1.28) both;
        }
        
        .thinking-text {
            color: #FFD700;
            font-weight: 500;
            font-family: 'Inter', sans-serif;
            font-size: 1rem;
        }
        
        .thinking-dots::after {
            content: '';
            animation: ellipsis 1.5s infinite;
            display: inline-block;
            width: 20px;
            margin-left: 4px;
        }
        
        /* Estilo para bloques de código */
        code {
            background: rgba(212, 175, 55, 0.15) !important;
            color: #FFE55C !important;
            padding: 0.2em 0.4em !important;
            border-radius: 0.25rem !important;
            font-family: 'Consolas', 'Monaco', monospace !important;
        }
        
        pre {
            background: rgba(18, 24, 40, 0.8) !important;
            border: 1px solid rgba(212, 175, 55, 0.2) !important;
            border-radius: 0.5rem !important;
            padding: 1rem !important;
        }
        
        /* Scrollbar personalizada */
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
        
        body {
            scrollbar-width: thin;
            scrollbar-color: #D4AF37 transparent;
        }
        
        /* Ajustes responsivos */
        @media (max-width: 640px) {
            .main .block-container {
                padding: 1rem 0.5rem;
            }
            
            .stChatMessage {
                margin-bottom: 1rem !important;
            }
            
            .stChatMessage[data-testid="StChatMessage"] .stChatMessageContent:has(div[data-testid="chatAvatarIcon-assistant"]),
            .stChatMessage[data-testid="StChatMessage"] .stChatMessageContent:has(div[data-testid="chatAvatarIcon-user"]) {
                padding: 1rem !important;
            }
            
            h1 {
                font-size: 1.5rem !important;
            }
        }
    </style>
    """, unsafe_allow_html=True)

def main():
    st.set_page_config(
        page_title="Midas Assistant",
        page_icon="🤖",
        layout="centered",
        initial_sidebar_state="collapsed"
    )

    # Cargar estilos personalizados
    load_custom_css()

    # Inicializar estado de la sesión
    initialize_session_state()

    # Encabezado simple
    st.title("🤖 Midas Assistant")
    st.markdown("""
    <p style="color: #D4AF37; margin-bottom: 2rem;">Asistente especializado en MIDAS (Multi-agent Intelligent Data Automation System)</p>
    """, unsafe_allow_html=True)
    
    # Verificar API Key
    if not API_KEY:
        st.warning("No se ha configurado una API Key en el archivo .env. El chat podría no funcionar correctamente.")
    
    # Mostrar historial de mensajes
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "🤖"):
            if message["role"] == "assistant":
                st.markdown(format_response(message["content"]))
            else:
                st.markdown(message["content"])

    # Entrada del usuario
    if prompt := st.chat_input("¿En qué puedo ayudarte con MIDAS?"):
        # Verificar API Key antes de procesar
        if not API_KEY:
            st.error("No se puede procesar la solicitud sin una API Key válida en el archivo .env.")
            return
            
        # Agregar mensaje del usuario al chat
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)
        
        # Agregar mensaje a la sesión
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Obtener y mostrar respuesta
        chat_history = [{"role": msg["role"], "content": msg["content"]} 
                        for msg in st.session_state.messages[:-1]]
        
        response = get_streaming_response(prompt, chat_history)
        
        # Agregar respuesta a la sesión
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()