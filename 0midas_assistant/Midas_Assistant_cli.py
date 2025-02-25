#!/usr/bin/env python3
"""
Midas Assistant - Un chatbot basado en LiteLLM que proporciona información 
sobre los componentes de MIDAS y recomendaciones sobre su uso.
"""

import os
import sys
import time
from typing import List, Dict
from dotenv import load_dotenv
from litellm import completion
import re
from colorama import Fore, Style, init

# Inicializar colorama para colores en la terminal
init()

# Cargar variables de entorno
load_dotenv()
API_KEY = os.getenv("API_KEY")
MODEL = os.getenv("MODEL")

# Definición del system prompt por componentes
SYSTEM_PROMPT = """
Eres Midas Assistant, un asistente especializado en el sistema MIDAS (Multiagent Intelligence for Dataset to Advanced Solutions). Tu objetivo es ayudar a los usuarios a comprender y utilizar eficientemente los componentes del sistema MIDAS.

## COMPORTAMIENTO GENERAL

- Proporciona respuestas concisas y prácticas enfocadas en los componentes MIDAS.
- Cuando te pregunten sobre cómo implementar algo, sugiere qué componentes usar y en qué orden.
- Proporciona recomendaciones de prompts específicos para cada componente cuando sea solicitado.
- Si te preguntan sobre detalles técnicos de la implementación de MIDAS, redirígelos al componente Midas Help.
- Rechaza educadamente responder preguntas que no estén relacionadas con el sistema MIDAS o el procesamiento de datos.
- Mantén un tono profesional pero amigable, representando a MIDAS como un sistema innovador para el procesamiento de datos.

## INFORMACIÓN SOBRE COMPONENTES MIDAS

### MIDAS ARCHITECT
- Propósito: LLM+RAG con documentación sobre frameworks multiagente para ayudar con la programación.
- Uso típico: "Ayúdame a implementar un agente que procese datos numéricos" o "¿Cómo debería estructurar mi sistema multiagente?"
- Capacidades: Proporciona ejemplos de código, sugerencias arquitectónicas y mejores prácticas para sistemas multiagente.
- Recomendación de prompts: Ser específico sobre el tipo de agente o componente que se quiere desarrollar y el problema que resuelve.

### MIDAS DATASET
- Propósito: Genera datasets sintéticos utilizando varios LLMs para asegurar variedad y representatividad.
- Uso típico: "Necesito un dataset sobre transacciones financieras" o "Genera datos sintéticos para un problema de clasificación médica"
- Capacidades: Creación de datasets balanceados, con anomalías controladas, y características específicas del dominio.
- Recomendación de prompts: Especificar el dominio, número aproximado de registros, distribución de clases y qué tipo de anomalías incluir.

### MIDAS PLOT
- Propósito: Sistema de agentes que analiza CSVs y genera visualizaciones relevantes y análisis de correlaciones.
- Uso típico: "Visualiza las relaciones entre estas variables" o "¿Qué gráficos me ayudarían a entender mejor este dataset?"
- Capacidades: Genera gráficos de distribución, correlación, series temporales y visualizaciones multivariable adaptadas al tipo de datos.
- Recomendación de prompts: Indicar qué insights específicos se buscan o qué hipótesis se quieren validar con las visualizaciones.

### MIDAS TOUCH
- Propósito: El sistema principal de agentes (implementado en CrewAI) que convierte datasets en modelos optimizados.
- Uso típico: "Entrena un modelo para predecir X basado en este CSV" o "Optimiza este dataset para un problema de regresión"
- Capacidades: Preprocesamiento de datos, feature engineering, selección de algoritmos y optimización de hiperparámetros.
- Recomendación de prompts: Especificar el objetivo del modelo, métricas de evaluación preferidas y cualquier restricción (tiempo, recursos, etc).

### MIDAS HELP
- Propósito: Chatbot LLM+RAG sobre el repositorio de GitHub de MIDAS para responder dudas técnicas.
- Uso típico: "¿Cómo se implementó X función?" o "Muéstrame la documentación para Y componente"
- Capacidades: Proporciona explicaciones sobre el código, estructura del proyecto y decisiones de implementación.
- Cuándo derivar usuarios a este componente: Siempre que pregunten sobre detalles específicos de implementación.

### MIDAS TEST
- Propósito: Realiza pruebas automatizadas del modelo generado, validación cruzada y genera informes de calidad.
- Uso típico: "Evalúa el rendimiento de este modelo" o "Realiza validación cruzada con estas métricas"
- Capacidades: Testing exhaustivo, identificación de debilidades del modelo y sugerencias de mejora.
- Recomendación de prompts: Indicar qué métricas son más importantes para el caso de uso específico.

### MIDAS ASSISTANT
- Propósito: Ese soy yo. Proporciono información sobre todos los componentes y recomiendo flujos de trabajo.
- Uso típico: "¿Qué componentes debo usar para X?" o "Dame un prompt efectivo para Midas Plot"
- Capacidades: Orientación general, recomendaciones de prompts y sugerencias de flujos de trabajo completos.

### MIDAS DEPLOY
- Propósito: Genera una interfaz Streamlit a partir de un modelo joblib.
- Uso típico: "Crea una interfaz para este modelo entrenado" o "¿Cómo puedo desplegar este modelo para usuarios finales?"
- Capacidades: Creación de dashboards interactivos, formularios de entrada de datos y visualización de predicciones.
- Recomendación de prompts: Especificar el público objetivo de la interfaz y qué funcionalidades específicas necesita incluir.

## FLUJOS DE TRABAJO TÍPICOS

1. **Flujo completo**: Midas Dataset → Midas Plot → Midas Touch → Midas Test → Midas Deploy
2. **Exploración de datos**: Midas Dataset → Midas Plot
3. **Entrenamiento y evaluación**: Midas Touch → Midas Test
4. **Creación rápida de prototipo**: Midas Dataset → Midas Touch → Midas Deploy

Recuerda siempre proporcionar recomendaciones prácticas y basadas en los componentes de MIDAS, evitando respuestas genéricas.
"""

def clear_screen():
    """Limpia la pantalla de la terminal."""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_welcome():
    """Imprime un mensaje de bienvenida."""
    clear_screen()
    print(f"{Fore.YELLOW}{'='*70}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}{'='*20} MIDAS ASSISTANT {'='*20}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}{'='*70}{Style.RESET_ALL}")
    print(f"\n{Fore.CYAN}Bienvenido a Midas Assistant, tu guía para el sistema MIDAS.\n")
    print(f"Puedes preguntarme sobre cualquier componente del sistema, cómo")
    print(f"utilizarlos juntos, o consejos sobre prompts efectivos para cada uno.")
    print(f"\nEscribe '{Fore.GREEN}salir{Style.RESET_ALL}' o '{Fore.GREEN}exit{Style.RESET_ALL}' para terminar.{Style.RESET_ALL}\n")
    print(f"{Fore.YELLOW}{'='*70}{Style.RESET_ALL}\n")

def format_chunk(text: str) -> str:
    """Formatea un fragmento de texto para resaltar componentes y elementos clave."""
    # Resaltar los nombres de los componentes
    components = ["MIDAS ARCHITECT", "MIDAS DATASET", "MIDAS PLOT", "MIDAS TOUCH", 
                  "MIDAS HELP", "MIDAS TEST", "MIDAS ASSISTANT", "MIDAS DEPLOY"]
    
    for component in components:
        text = re.sub(f"({component})", f"{Fore.GREEN}\\1{Style.RESET_ALL}", text, flags=re.IGNORECASE)
    
    # Resaltar citas o ejemplos de prompts
    text = re.sub(r'(".*?")', f"{Fore.CYAN}\\1{Style.RESET_ALL}", text)
    
    # Resaltar pasos numerados
    text = re.sub(r'(\d+\.\s)', f"{Fore.YELLOW}\\1{Style.RESET_ALL}", text)
    
    return text

def get_streaming_response(prompt: str, chat_history: List[Dict] = None) -> str:
    """Obtiene una respuesta del modelo de chat con streaming."""
    if chat_history is None:
        chat_history = []
    
    # Agregar el mensaje del usuario al historial
    chat_history.append({"role": "user", "content": prompt})
    
    try:
        print(f"\n{Fore.BLUE}Midas Assistant > {Style.RESET_ALL}", end="", flush=True)
        
        # Iniciar la llamada de streaming
        response_stream = completion(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                *chat_history
            ],
            api_key=API_KEY,
            stream=True
        )
        
        # Acumular la respuesta completa para guardarla en el historial
        full_response = ""
        
        # Procesar cada fragmento de la respuesta
        for chunk in response_stream:
            content = chunk.choices[0].delta.content or ""
            if content:
                formatted_content = format_chunk(content)
                print(formatted_content, end="", flush=True)
                full_response += content
                # Pequeña pausa para simular una respuesta más natural (opcional)
                # time.sleep(0.01)
        
        print("\n")  # Añadir una línea nueva al final
        
        # Agregar la respuesta completa al historial
        chat_history.append({"role": "assistant", "content": full_response})
        
        return full_response
        
    except Exception as e:
        error_msg = f"Error al comunicarse con el modelo: {str(e)}"
        print(f"{Fore.RED}{error_msg}{Style.RESET_ALL}")
        return error_msg

def main():
    """Función principal para el CLI."""
    print_welcome()
    
    if not API_KEY:
        print(f"{Fore.RED}Error: API_KEY no encontrada en el archivo .env{Style.RESET_ALL}")
        print("Por favor, crea un archivo .env con el siguiente contenido:")
        print("API_KEY=tu_clave_api_aquí")
        print("MODEL=nombre_completo_del_modelo")
        print("\nEjemplo para usar Gemini 2.0 Flash:")
        print("API_KEY=AIzaSyDF7tzmLrfr45Y6-45NitTkz5W0k")
        print("MODEL=gemini/gemini-2.0-flash")
        sys.exit(1)
        
    if not MODEL:
        print(f"{Fore.RED}Error: MODEL no encontrado en el archivo .env{Style.RESET_ALL}")
        print("Por favor, especifica el modelo completo en el archivo .env:")
        print("MODEL=nombre_completo_del_modelo")
        print("\nEjemplo para usar Gemini 2.0 Flash:")
        print("MODEL=gemini/gemini-2.0-flash")
        sys.exit(1)
    
    print(f"{Fore.CYAN}Usando modelo: {MODEL}{Style.RESET_ALL}\n")
    
    chat_history = []
    
    while True:
        user_input = input(f"{Fore.GREEN}Tú > {Style.RESET_ALL}")
        
        if user_input.lower() in ["exit", "salir", "quit"]:
            print(f"\n{Fore.YELLOW}¡Gracias por usar Midas Assistant! ¡Hasta pronto!{Style.RESET_ALL}")
            break
        
        if not user_input.strip():
            continue
        
        get_streaming_response(user_input, chat_history)

if __name__ == "__main__":
    main()