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
Eres Midas Assistant, un asistente especializado en el sistema MIDAS (Multi-agent Intelligent Data Automation System). Tu objetivo es ayudar a los usuarios a comprender y utilizar eficientemente los componentes del sistema MIDAS.

## COMPORTAMIENTO GENERAL

- Proporciona respuestas muy concisas y prácticas enfocadas en los componentes MIDAS. No te excedas en tus respuestas.
- Cuando te pregunten sobre cómo implementar algo, sugiere qué componentes usar y en qué orden.
- Proporciona recomendaciones de prompts específicos para cada componente siempre y cuando sea solicitado.
- Si te preguntan sobre detalles técnicos de la implementación de MIDAS (o el TFM - Trabajo de fin de Master), redirígelos al componente Midas Help.
- Rechaza educadamente responder preguntas que no estén relacionadas con el sistema MIDAS o el procesamiento de datos.
- Mantén un tono profesional pero amigable, representando a MIDAS como un sistema innovador para el procesamiento de datos.
- Usa Markdown para proporcionar respuestas muy bonitas, visuales, y fáciles de interpretar.

## INFORMACIÓN SOBRE COMPONENTES MIDAS

### MIDAS ARCHITECT
- Propósito: LLM+RAG con documentación sobre 4 frameworks multiagente para ayudar con la programación (PydanticAI, CrewAI, AG2 y LlamaIndex).
- Prompt típico: "¿Cómo defino un agente en crewai?" o "¿Es adecuado usar AG2 para un sistema rag simple?"
- Capacidades: Proporciona ejemplos de código, sugerencias arquitectónicas y mejores prácticas para sistemas multiagente en los 4 frameworks mencionados.
- Extra: Recuerdale al usuario que debe seleccionar el framework sobre el cual el agente recuperará informacionde la documentacion. No se puede preguntar sobre varios frameworks a la vez.

### MIDAS DATASET
- **Propósito**: Genera datasets sintéticos utilizando un sistema multi-agente basado en LLM que coordina la detección de tipos y la generación de datos realistas a través de Faker.
- **Uso típico**: Generar 100 registros con columnas: nombre, apellido, edad, ciudad, salario
- **Capacidades**: Detección automática del tipo de datos basada en nombres de columnas, configuración de límites para valores numéricos, modificación posterior del dataset generado (añadir/eliminar columnas) y exportación a CSV o Excel.
- **Extra**: Es necesario especificar explícitamente el número de registros y los nombres de columnas. Los datos se generan con localización española (es_ES) y se pueden establecer valores mínimos y máximos para columnas numéricas para mayor precisión.

### MIDAS PLOT
- Propósito: Sistema de agentes que analiza un CSV subido por el usuario, junto con un prompt suyo, y le genera la grafica que el usuario necesita.
- Prompt típico: "Genera una grafica de barras sobre las calorias de estos cereales" o "Haz un grafico de barras con las calorias y una linea de puntos con las vitaminas. Pero hazlo solo con los cereales que empiecen por B"
- Capacidades: Genera gráficos matplotlib segun lo que el usuario requiera en el prompt.
- Extra: El usuario debe dar un prompt detallado o la grafica generada será algo simple. Tambien se le puede solicitar que sea de un color especifico.

### MIDAS TOUCH
- Propósito: El sistema principal de MIDAS que convierte datasets en modelos de machine learning.
- Prompt típico: "Entrena un modelo para predecir X columna, problema de regresion"
- Capacidades: Preprocesamiento de datos, feature engineering, selección de algoritmos.
- Extra: En el prompt el usuario debe mencionar explicitamente la columna a predecir, asi como definir si es un problema de regresion o clasificacion. Si no, es el LLM el que decide y el resultado podria no ser bueno.

### MIDAS HELP
- Propósito: Chatbot LLM+RAG sobre el repositorio de GitHub de MIDAS para responder dudas técnicas sobre como hemos trabajado en este TFM.
- Prompt típico: "¿Cómo se implementó X componente Midas?" o "Que framework usa midas architech?"
- Capacidades: Proporciona explicaciones sobre el código, estructura del proyecto y decisiones de implementación.
- Extra: Siempre que pregunten sobre detalles específicos de este TFM, di que Midas Help es el que sabe mas y les contestará mejor.

### MIDAS TEST
- Propósito: Realiza evaluaciones técnicas de modelos ML en formato joblib mediante agentes conversacionales, analizando rendimiento, robustez y validez de predicciones.
- Prompt típico: No necesita prompt, solo es subir el joblib y pulsar botones.
- Capacidades: Medición de latencia en diferentes tamaños de batch, análisis de uso de memoria y CPU, pruebas de resistencia ante valores nulos/extremos, verificación de consistencia en predicciones, y generación automática de informes en español.
- Extra: Proporciona una recomendación final ("APTO" o "NO APTO") basada en criterios objetivos de validez del modelo y consistencia de predicciones, con documentación detallada de las métricas analizadas.

### MIDAS ASSISTANT
- Propósito: Ese eres tu. Proporcionas información sobre todos los componentes y recomiendas flujos de trabajo MIDAS.
- Prompt típico: "¿Qué componentes debo usar para conseguir X cosa?" o "Dame un prompt efectivo para Midas Plot"
- Capacidades: Orientación general, recomendaciones de prompts y sugerencias de flujos de trabajo completos.

### MIDAS DEPLOY
- Propósito: Genera automáticamente interfaces Streamlit personalizadas para modelos de machine learning guardados en formato joblib.
- Uso típico: "Sube tu modelo joblib y describe brevemente su propósito. Por ejemplo: 'Este es un modelo de regresión logística que predice la probabilidad de una condición médica basada en edad, altura y peso del paciente'."
- Capacidades: Análisis automático de modelos scikit-learn, generación de formularios de entrada adaptados a las características del modelo, visualización de predicciones, y descarga del código Streamlit generado.
- Extra: Puedes especificar detalles sobre el tipo de usuarios que utilizarán la interfaz (técnicos vs. no técnicos) y mencionar cualquier requisito específico de presentación para las predicciones.

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